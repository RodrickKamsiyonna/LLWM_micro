# Character-level latent-action language model

The compressor encodes text into causal memories. A Gaussian action encoder sees the current memory and **next compressor memory**. An action-conditioned predictor and vocabulary head reconstruct that next token. The predictor output also predicts the next compressor memory, supervised by a detached next-memory MSE target.

## Training

```text
h = compressor(input_tokens)
mu, log_std = action_encoder(detach(h), detach(next_memory))
action = mu + exp(log_std) * Normal(0, I)
features = predictor(h, action)       # predicted next memory; adaptive LayerNorm conditioning
logits = 15 * tanh(head(features) / 15)
kl = mean_valid_tokens_and_dimensions(KL(q(action) || Normal(0, I)))
kl_total = kl * action_dim
excess = relu(kl_total - target_kl)
loss = ce_weight * CE(logits, next_tokens) + kl_lambda * excess**2 + mse_weight * MSE(features, detach(next_memory))
```

CE trains the compressor, action encoder, predictor, and head. The previously requested stop-gradients remain at both action-encoder inputs; CE reaches the compressor through the direct predictor input. The head and predictor output are **not detached**. There is no separate compressor-head CE. MSE trains the predictor, action encoder, and current-memory compressor path; its next-memory target is detached. `--mse-weight` defaults to 1 and must be finite and nonnegative. W&B logs the unweighted MSE as `train/mse` and `val/mse`.

## Detached memory probe

A separate linear vocabulary head measures next-character information in the compressor's **current memory**, without updating that memory:

```python
probe_logits = memory_probe(memory.detach())   # m[t] -> x[t+1]
probe_ce = cross_entropy(capped(probe_logits), next_tokens, ignore_index=-1)
```

The existing predictor head and all original model objectives are preserved. This probe is an independent diagnostic readout, with its own parameters; it is not used by the action encoder, predictor, EQM, or planner. Probe initialization preserves the existing model's RNG stream.

Only `memory_probe` receives gradients from `probe_ce`. The trainer backpropagates `model_loss + probe_ce`, but logs `loss` and selects best checkpoints using the original model objective alone. The returned `WorldModel.forward()['loss']` excludes the diagnostic probe; callers training the probe should add `probe_ce` as the trainer does. Probe and main-model gradients are clipped independently, so a large probe gradient cannot change the main model's clipping coefficient. Both use AdamW and the existing learning-rate schedule.

W&B logs **`train/probe_ce`**, **`train/probe_accuracy`**, **`val/probe_ce`**, and **`val/probe_accuracy`**. Accuracy is the fraction of valid next characters predicted correctly. Ignored target positions do not contribute; entirely ignored batches return zero. Logits use the same smooth cap as the original head. Lower validation probe CE and higher validation accuracy indicate more next-character information accessible to this linear readout; they do not measure all information in the memory.

New checkpoints save the probe weights and optimizer moments. Resuming a pre-probe checkpoint restores existing model weights, optimizer moments, scaler and RNG streams, while initializing only the new diagnostic head and its fresh optimizer state. Partial or unrelated architecture mismatches remain errors. Existing best-validation scores remain comparable because the diagnostic loss is excluded from that score.

## KL budget

`kl` is the mean Gaussian KL over valid tokens and action dimensions. Therefore **`kl_total = kl * action_dim` is the mean total latent KL per valid token**, measured in nats. It is not the sum across the whole minibatch. To get that sum, multiply by the number of valid tokens as well.

The required `--target-kl` sets the budget; there is deliberately no default target. `--kl-lambda` defaults to 1 and weights the squared positive excess:

```python
excess = F.relu(kl_total - target_kl)
loss = ce_weight * ce + kl_lambda * excess.square() + mse_weight * mse
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

CE, KL, and MSE ignore target index `-1`; an entirely ignored batch returns finite zero losses. Ignored targets do not consume the KL budget. The final target token clamps ignored indices to zero for the extra memory-target encoding. Training uses FP16 autocast/gradient scaling on CUDA and FP32 on CPU. CE computation and its gradient paths are unchanged.

## Optional equilibrium matching (EQM)

Enable with `--eqm-weight 1 --eqm-lambda 1 --eqm-chunk-size 128` (the weights are configurable examples). The default weight is **0**, which completely skips EQM computation and preserves the current losses. `--eqm-lambda` defaults to 1 and scales the target gradient field; it is separate from the outer loss coefficient `--eqm-weight`. Both must be finite and nonnegative.

For each valid transition, EQM uses the sampled training action as a detached target:

```python
gamma = Uniform(0, 1)
eps = Normal(0, I)
noisy_action = (gamma * action.detach() + (1 - gamma) * eps).detach().requires_grad_(True)
energy = CE(head(predictor(memory, noisy_action)), next_token, reduction='sum')
grad_energy = autograd.grad(energy, noisy_action, create_graph=True)[0]
target_gradient = (eps - action.detach()) * eqm_lambda * (1 - gamma)
eqm = mean((grad_energy - target_gradient)**2)
loss = ce_weight * ce + mse_weight * mse + kl_penalty + eqm_weight * eqm
```

The noisy action is a fresh detached gradient input: EQM does not train the action encoder through either the interpolation or target field. **Memory is not detached.** EQM trains the compressor through memory, plus the predictor and LLM head. This additional training objective leaves memory-space planning and its inverse-dynamics consistency regularizer unchanged.

`--eqm-chunk-size` counts valid memory/action/token transitions, flattened across batch and sequence. The default 128 processes at most 128 transitions in each EQM predictor call; 0 uses all valid transitions in one chunk. All valid transitions contribute, including the final partial chunk, with normalization by the total valid-token count times action dimension. Ignored tokens (`-1`) are excluded. No transitions are subsampled. Noise is sampled before chunking so chunk size does not change the sampled objective.

EQM restores the custom-autograd recomputation strategy from commit `327fe65`. Forward computes only the first action derivative and releases each chunk graph. Backward rebuilds one chunk with `create_graph=True`, computes its parameter and memory gradients, accumulates those tensors, and releases the graph before constructing the next chunk. It does not use the checkpoint-based implementation introduced in `b7b76be`.

Local memory leaves inside custom backward isolate recomputation; their gradients are explicitly returned to the original attached compressor memory. This preserves compressor gradients while the noisy action and target action stay detached. AMP's outer loss scale is applied only after differentiating the chunk, never to the inner CE field. Smaller chunks reduce per-chunk activation storage but add overhead. The shared compressor and ordinary CE/MSE graphs still consume memory. Validation computes first action derivatives without keeping a parameter-update graph.

In an isolated CPU benchmark (256 transitions, width 64, three predictor layers, action dimension 16, chunks of 32), full EQM and recomputation both returned loss 0.6695118. Peak unique storage retained by autograd saves fell from 5,857,028 to 1,211,528 bytes (about 4.8x lower); time rose from about 44 to 132 ms. This saved-storage proxy is not process RSS or CUDA allocator memory. Measure actual T4 peaks with:

```bash
python benchmark_eqm.py --device cuda --tokens 4096 --width 256 \
  --action-dim 32 --layers 3 --chunk-size 128
```

W&B logs raw `train/eqm`, `val/eqm` and weighted `train/eqm_penalty`, `val/eqm_penalty`. These settings are saved in checkpoints, and changing EQM weight or field scale resets best validation loss.

## Dataset selection: Shakespeare or WikiText-2 (character-level)

The default `--dataset shakespeare` downloads Tiny Shakespeare. To train on the **small WikiText-2 raw dataset** instead, use `--dataset wikitext`. The dataset is [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext), configuration `wikitext-2-raw-v1` (the raw *train* split). The downloader saves its text to `data/wikitext.txt`, which is reused on subsequent runs. No word/subword tokenizer is used: `train.py` constructs the vocabulary from **individual Unicode characters**, and each character is one token. A fixed 90% / 10% character split preserves this project's existing validation procedure.

To download separately:

```bash
pip install -r requirements.txt
python prepare.py --dataset wikitext
```

Or simply start training; the rank-zero training process downloads missing data automatically, and other distributed workers wait for completion:

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --dataset wikitext --target-kl 5.0 \
  --steps 20000 --batch-size 16 --accumulation 4 \
  --out checkpoints/wikitext
```

Use `--dataset shakespeare` for the original corpus (default), or `--data path/to/your.txt` to train on an existing UTF-8 file without downloading. WikiText uses a default prompt of `The ` rather than `ROMEO:\\n`, unless you explicitly pass `--prompt`. Changing the corpus or vocabulary is **not** checkpoint-compatible; use a new output directory and train from scratch. For a WikiText experiment, the actual words in a text sample are still just sequences of characters to the model.

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
  --ce-weight 1 --mse-weight 1 --target-kl 5.0 --kl-lambda 1 --kl-steps 1000 \
  --plan-horizon 500 --plan-iterations 100 --plan-lr 0.05 --plan-consistency-weight 1 \
  --goal ' .' --run-name token-action
```

The target 5.0 and 1,000-update ramp above are examples: choose your own `--target-kl` and `--kl-steps`. Other defaults are shown above. Compressor/predictor feed-forward dimension is `4*width`; the action encoder has two linear layers, hidden dimension `2*width`, and outputs action mean/log standard deviation. Dropout defaults to 0.1. Width must be divisible by the head count, with an even per-head dimension for rotary positions. The model uses rotary positions and a sliding causal attention window of `--context` tokens. Positions can extend past the training context during planning.

Each GPU samples independent random windows. Two GPUs with batch size 16 and four accumulation microsteps give 128 sequences per optimizer update. AdamW uses weight decay 0.01; learning rate warms up over 200 steps, then follows cosine decay to 10% of the peak. `--eval-every` and `--save-every` default to 500 optimizer steps. Validation/planning also run at the final step. Overflow attempts are retried without counting as completed optimizer updates.

The dataset is [Karpathy's Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare), split contiguously into 90% training and 10% validation. `prepare.py` explicitly downloads it; `--data` can point to another UTF-8 text file.

## Planning in memory space

Compress the prompt once, then optimize one initially Gaussian action per generated token:

```text
h = compressor(prompt)[-1]  # once, without gradients
for t in range(horizon):
    memories_before[t] = h
    h = predictor(h, actions[t])
    memories[t] = h
logits = capped_head(memories)
mu[t], _ = action_encoder(memories_before[t], memories[t])  # per transition
consistency = mean((actions - mu)**2)
planning_loss = CE(logits[-len(goal):], goal) + consistency_weight * consistency
```

Each predicted memory is the input to the next transition. The same LLM head used during training decodes the token added by each transition. A 500-token plan has 500 actions and 500 memory transitions; the last predicted memory decodes the 500th token. There are no soft-token embeddings, compressor feedback calls, or KV-cache updates during rollout. The prompt is truncated to its final context-length tokens.

Goal characters are targets only: they are never fed into the predictor or action encoder. The goal can be multiple characters or lines and must fit in the horizon and vocabulary. Planning combines goal-suffix CE with inverse-dynamics consistency: each optimized action is compared by MSE to the action encoder's Gaussian mean for its own predicted `(current_memory, next_memory)` pair. The action vectors are not averaged across trajectories, and no action is sampled for this penalty. MSE averages the squared elementwise differences across batch, horizon, and action dimensions. `--plan-consistency-weight` defaults to 1; 0 disables the penalty and skips the inverse encoder computation. Inverse consistency is a planning-only regularizer; training uses CE, memory MSE, the KL budget, and optional EQM. It encourages consistency with learned transitions, without guaranteeing valid trajectories. All model parameters are frozen and restored afterward; gradients traverse the complete memory trajectory and both inverse-encoder memory inputs to optimize only the actions. The inverse encoder inputs are detached only during training, not during planning.

Planning uses Adam for 100 iterations at learning rate 0.05 by default. The initialized candidate and final update are included when selecting the best combined planning objective. The reported goal CE remains separate from that combined loss. After optimization, the best memory trajectory is decoded with the LLM head using argmax. Exact match checks that decoded suffix; the goal is never appended to the output. Low goal loss does not guarantee an exact match or coherent intermediate text. Long memory rollouts can accumulate prediction error; one-step MSE does not guarantee long-horizon accuracy. The obsolete `--plan-temperature` option has been removed because no soft-token feedback is used.

Training expects contiguous shifted windows (`idx=x[:-1]`, `targets=x[1:]`). Next-memory targets reuse the shifted compressor output and encode the final target once with the training prefix cache, covering all positions without a second full compressor pass. Targets are detached, while the current-memory path stays differentiable. KV caching remains available for compressor encoding but is unnecessary during memory-space planning.

## Logging, checkpoints, and compatibility

W&B is online by default (`wandb login`). `--wandb-mode offline` saves locally; `disabled` turns logging off. Rank zero logs `loss`, `ce`, per-dimension `kl`, `kl_total`, `kl_excess`, weighted `kl_penalty`, and next-memory `mse` under both `train/` and `val/`, plus learning rate, gradient norm, and planning text/history. Planning logs `plan/inverse_consistency` (raw MSE), `plan/consistency_penalty` (weighted MSE), `plan/goal_ce`, and `plan/best_loss` (their sum for the selected trajectory). Optimization history and planning JSON include both components. The weighted penalty is exactly `kl_lambda * kl_excess**2`, the KL contribution added to total loss. `--project` and `--run-name` customize the run.

The new default output directory is `checkpoints/token_action`, preserving older experiment artifacts. Numbered checkpoints retain model/optimizer/scaler, configuration, vocabulary, dataset hash, step, best validation loss, W&B ID, and per-rank Python/NumPy/Torch/CUDA RNG state. `latest.pt` and `best.pt` point to relevant files. Planning results are also written as `plan_STEP.json`.

**Old memory-MSE/learned-position checkpoints are incompatible; start a fresh run.** Resume new checkpoints with matching arguments and `--resume checkpoints/token_action/latest.pt`. Keep world size and total steps unchanged for RNG and learning-rate continuity. Only load trusted checkpoint files. Changing the objective, target, MSE weight, schedule, or penalty strength resets the stored best validation loss for comparison. Existing token-action architecture checkpoints remain structurally loadable, but their action encoder learned token-embedding inputs. Retrain or fine-tune it on memory pairs before relying on inverse-consistency planning. The changed objective resets best validation loss. The CLI requires an explicit target.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests cover next-token CE and per-dimension KL, analytical KL totals and budget gradients, coefficient ramp endpoints, ignored targets, main-model CE gradients and detached-probe isolation, detached action-encoder inputs, full-versus-cached sliding attention, next-memory target alignment and stop-gradients, and autoregressive memory-rollout/action gradients. Planning tests assert that the compressor is called exactly once and goal tokens never enter the model. CPU mixed-precision gradients are checked as well.

All 41 tests passed, including detached-probe gradient isolation, independent gradient clipping, ignored targets, old/new checkpoint restoration, and exclusion of the probe from planning. EQM regression tests retain exact loss/parameter-gradient comparisons, outer gradient scaling and saved-memory checks. A two-process CPU/Gloo smoke run completed training with gradient accumulation and chunked EQM, validation, planning and checkpoint saving. Its main-model weights, AdamW moments and best validation score exactly matched the archived run without the probe. Offline W&B recorded probe CE and accuracy for train and validation. Actual T4 execution and resolution of the reported GPU training OOM remain unverified locally.

Local verification uses PyTorch 2.2.2/NumPy 1.26.4/W&B 0.18.7 on an Intel Mac. The GPU requirements target PyTorch 2.4+. Actual T4 FP16/NCCL performance and peak GPU memory remain unverified locally.
