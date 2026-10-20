# Character-level latent-action language model

The compressor encodes text into causal memories. A Gaussian action encoder sees the current memory and **next-token embedding**. An action-conditioned predictor and vocabulary head reconstruct that next token. Predictor features are not next-memory targets: there is no memory MSE loss.

## Training

```text
h = compressor(input_tokens)
mu, log_std = action_encoder(detach(h), detach(embedding(next_tokens)))
action = mu + exp(log_std) * Normal(0, I)
features = predictor(h, action)       # adaptive LayerNorm conditioning
logits = 15 * tanh(head(features) / 15)
kl = mean_valid_tokens_and_dimensions(KL(q(action) || Normal(0, I)))
kl_total = kl * action_dim
excess = relu(kl_total - target_kl)
loss = ce_weight * CE(logits, next_tokens) + kl_lambda * excess**2
```

CE trains the compressor, action encoder, predictor, and head. The previously requested stop-gradients remain at both action-encoder inputs; CE reaches the compressor through the direct predictor input. The head and predictor output are **not detached**. There is no separate compressor-head CE and no predictor-memory MSE.

## KL budget

`kl` is the mean Gaussian KL over valid tokens and action dimensions. Therefore **`kl_total = kl * action_dim` is the mean total latent KL per valid token**, measured in nats. It is not the sum across the whole minibatch. To get that sum, multiply by the number of valid tokens as well.

The required `--target-kl` sets the budget; there is deliberately no default target. `--kl-lambda` defaults to 1 and weights the squared positive excess:

```python
excess = F.relu(kl_total - target_kl)
loss = ce_weight * ce + kl_lambda * excess.square()
```

This is a one-sided upper-budget penalty: below or at the target it is zero, so it does not push KL up toward the target. Above it, the penalty grows quadratically. Setting `--kl-lambda 0` disables the penalty while retaining raw KL diagnostics. Target and coefficient must be finite and nonnegative.

### KL coefficient warmup

`--kl-steps N` linearly ramps the training coefficient to the configured `--kl-lambda`. For `N >= 2`, optimizer update 1 uses zero, update N reaches the full coefficient, and later updates keep that value:

```python
effective_kl_lambda = kl_lambda * min(1.0, max(0.0, (step - 1) / (kl_steps - 1)))
```

`--kl-steps 0` is the default and disables the ramp, preserving immediate full strength. A one-update ramp (`--kl-steps 1`) reaches full strength on its first update. The ramp advances per successful optimizer update, not per microbatch or GPU; FP16 overflow retries reuse the same coefficient. Resuming uses the saved global step, so warmup does not restart. Keep `--kl-steps` and `--kl-lambda` unchanged when resuming the same schedule.

W&B logs the effective coefficient as `train/kl_lambda`. Training `kl_penalty` uses that coefficient. Validation always uses the full configured coefficient and logs `val/kl_lambda`, keeping validation loss comparable throughout warmup. Checkpoints include the ramp configuration and the coefficient used at their saved step.

The budget applies to the mean valid-token KL of each forward microbatch. DDP and gradient accumulation average those microbatch objectives. Since the penalty is nonlinear, the average of local penalties need not equal a penalty recomputed from the globally averaged logged KL. This implementation follows the formula on each forward call.

CE and KL ignore target index `-1`; an entirely ignored batch returns finite zero losses. Ignored targets do not consume the KL budget. Next-token embeddings clamp ignored indices to zero before lookup. Training uses FP16 autocast/gradient scaling on CUDA and FP32 on CPU. CE computation and its gradient paths are unchanged.

## Run on two T4 GPUs

Use Python 3.10+ and a CUDA-enabled PyTorch installation appropriate for your system:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prepare.py
wandb login
torchrun --standalone --nproc_per_node=2 train.py \
  --steps 20000 --context 256 --batch-size 16 --accumulation 4 \
  --width 256 --heads 4 --compressor-layers 4 --predictor-layers 3 \
  --action-dim 32 --lr 3e-4 --warmup 200 \
  --ce-weight 1 --target-kl 5.0 --kl-lambda 1 --kl-steps 1000 \
  --plan-horizon 500 --plan-iterations 100 --plan-lr 0.05 \
  --plan-temperature 1 --goal ' .' --run-name token-action
```

The target 5.0 and 1,000-update ramp above are examples: choose your own `--target-kl` and `--kl-steps`. Other defaults are shown above. Compressor/predictor feed-forward dimension is `4*width`; the action encoder has two linear layers, hidden dimension `2*width`, and outputs action mean/log standard deviation. Dropout defaults to 0.1. Width must be divisible by the head count, with an even per-head dimension for rotary positions. The model uses rotary positions and a sliding causal attention window of `--context` tokens. Positions can extend past the training context during planning.

Each GPU samples independent random windows. Two GPUs with batch size 16 and four accumulation microsteps give 128 sequences per optimizer update. AdamW uses weight decay 0.01; learning rate warms up over 200 steps, then follows cosine decay to 10% of the peak. `--eval-every` and `--save-every` default to 500 optimizer steps. Validation/planning also run at the final step. Overflow attempts are retried without counting as completed optimizer updates.

The dataset is [Karpathy's Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare), split contiguously into 90% training and 10% validation. `prepare.py` explicitly downloads it; `--data` can point to another UTF-8 text file.

## Planning with soft embeddings and a differentiable KV cache

Compress the prompt once, then optimize one initially Gaussian action per generated token:

```text
h, prefix_cache = compressor(prompt)  # computed once, without gradients
cache = prefix_cache
for t in range(horizon):
    logits[t] = head(predictor(h, actions[t]))  # same capped logits as training
    if t + 1 < horizon:
        probabilities = softmax(logits[t] / temperature)
        soft_embedding = probabilities @ token_embedding_table
        h, cache = compressor(soft_embedding, cache)
planning_loss = CE(logits[-len(goal):], goal)
```

A soft embedding is the probability-weighted expected token embedding, not a raw-logit-weighted vector or a random sample. `--plan-temperature` controls the softmax temperature and defaults to 1. Every subsequent memory comes from re-encoding the previous soft token through the compressor. **Predictor features are never fed back as memory.** A 500-token plan has 500 actions/predictions and 499 soft-feedback compressor updates after the prompt.

The prompt is truncated to the final context-length tokens. Its prefix cache is reused across optimization iterations. Generated-token K/V caches remain differentiable: detaching them would cut goal gradients to earlier actions. Cache updates are out of place, so the original prefix cache is never mutated. Each layer retains at most `context-1` previous K/V positions; adding the next token forms a context-length attention window. Single-query decoding attends to all retained past keys, rather than using an incorrectly aligned causal mask. Rotary positions continue increasing as the window slides.

Caching saves repeated prefix computation. It does not make planning's backward graph constant-memory: generated states/caches must remain connected across the horizon. Attention uses PyTorch scaled-dot-product attention. All model parameters are frozen during planning and restored afterward; only actions are optimized. No action encoder, goal embeddings, action regularizer, or memory MSE is used in planning. Goal characters are CE targets only. There is no final unnecessary soft-feedback update after the last prediction.

Planning uses Adam for 100 iterations at learning rate 0.05 by default. The initialized candidate and final update are included when selecting the best goal loss. Logged text is the argmax decoding of the **soft-feedback trajectory**; feeding those hard argmax tokens back could produce a different trajectory. A low soft planning loss does not guarantee successful discrete generation or coherent intermediate text. Exact match checks the actual decoded soft-rollout suffix, not appended target text.

## Logging, checkpoints, and compatibility

W&B is online by default (`wandb login`). `--wandb-mode offline` saves locally; `disabled` turns logging off. Rank zero logs `loss`, `ce`, per-dimension `kl`, `kl_total`, `kl_excess`, and weighted `kl_penalty` under both `train/` and `val/`, plus learning rate, gradient norm, and planning text/history. The weighted penalty is exactly `kl_lambda * kl_excess**2`, the KL contribution added to total loss. `--project` and `--run-name` customize the run. No obsolete MSE metric is logged.

The new default output directory is `checkpoints/token_action`, preserving older experiment artifacts. Numbered checkpoints retain model/optimizer/scaler, configuration, vocabulary, dataset hash, step, best validation loss, W&B ID, and per-rank Python/NumPy/Torch/CUDA RNG state. `latest.pt` and `best.pt` point to relevant files. Planning results are also written as `plan_STEP.json`.

**Old memory-MSE/learned-position checkpoints are incompatible; start a fresh run.** Resume new checkpoints with matching arguments and `--resume checkpoints/token_action/latest.pt`. Keep world size and total steps unchanged for RNG and learning-rate continuity. Only load trusted checkpoint files. Changing the objective, target, or penalty strength resets the stored best validation loss for comparison. Existing token-action architecture checkpoints remain loadable; the new CLI requires an explicit target.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests cover next-token CE and per-dimension KL, analytical KL totals and budget gradients, below/at/above-budget behavior, ignored targets, required CLI target validation, all-module CE gradients, detached action-encoder inputs, full-versus-cached sliding attention, cache immutability, and soft-rollout/action-gradient equivalence to full recomputation. CPU mixed-precision gradients are checked as well.

All 23 tests passed. A two-process CPU/Gloo run completed training, validation, cached planning, and checkpoint saving. Offline W&B recorded the budget losses and a KL coefficient ramp of 0, 0.5, and 1, while validation used the full coefficient throughout. Resuming from the first checkpoint exactly reproduced the uninterrupted model, optimizer, scaler, RNG state, validation result, and planning outputs.

Local verification uses PyTorch 2.2.2/NumPy 1.26.4/W&B 0.18.7 on an Intel Mac. The GPU requirements target PyTorch 2.4+. Actual T4 FP16/NCCL performance and peak GPU memory remain unverified locally.
