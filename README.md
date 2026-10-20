# LLM world model: character-level Shakespeare

Four modules: a causal Transformer compressor, Gaussian action encoder, action-conditioned memory predictor with adaptive LayerNorm, and a vocabulary head. The predictor processes each training transition independently; planning recursively uses its own predicted memories.

For characters `x[0:T+1]`, the compressor produces `m[0:T+1]`. Each `m[t]` has access only to `x[:t+1]`. Training uses:

```
q(a[t] | stop_gradient(m[t]), stop_gradient(m[t+1])) = Normal(mu, exp(log_std))
a[t] = mu + exp(log_std) * epsilon
predicted_m[t+1] = predictor(m[t], a[t])
logits[t+1] = head(m[t])  # attached compressor memory
loss = ce_weight * CE(logits[t+1], x[t+1])
     + mse_weight * MSE(predicted_m[t+1], stop_gradient(m[t+1]))
     + kl_weight * KL(q(a[t]) || Normal(0, I))
```

The head receives **attached current compressor memory** and predicts the next character. CE directly trains both the compressor (including token/position embeddings) and the head; it does not update the predictor or action encoder. The action encoder receives detached current and next compressor memories. The predictor still takes attached current memory and the sampled action, and learns next memory through MSE against the detached next-memory target. The action encoder learns through MSE and action KL. Planning keeps gradients through predicted memories to optimize actions.

Default loss weights: CE=1, MSE=1, KL=0.001, EQM=0. The active objective is CE + memory MSE + action KL. `--ce-weight` controls compressor next-token CE. EQM is no longer used in this workflow and is skipped entirely by default in both the model API and training CLI. Its implementation is retained unchanged for explicit legacy experiments. No new flag is needed to disable it; `--disable-eqm` remains available.

## Legacy equilibrium matching (disabled by default)

```text
gamma ~ Uniform(0, 1); eps ~ Normal(0, I)
a_gamma = gamma * detach(action) + (1 - gamma) * eps
energy = sum_token_CE(head(predictor(memory, a_gamma)), next_tokens)
grad_energy = d energy / d a_gamma
desired = (eps - detach(action)) * eqm_lambda * (1 - gamma)
eqm = mean((grad_energy - desired)**2)
```

`--eqm-weight` now defaults to 0: nothing in this section is computed in the active workflow. The unchanged legacy implementation can be explicitly enabled with a positive weight. `--eqm-lambda` defaults to 1 and controls the target gradient field, while the weight controls the contribution to total loss. `--eqm-chunk-size` defaults to 128 transitions. `--disable-eqm` overrides any supplied weight. Disabled EQM logs zero and does not perform noisy predictor passes or higher-order differentiation.

`llwm/eqm.py` uses exact chunked recomputation. Forward evaluates the loss with a first action derivative and releases each chunk's graph. A custom backward recomputes one chunk at a time with `create_graph=True`, accumulating mixed second derivatives into parameter and memory gradients. The same gamma/noise samples are reused. This avoids retaining a full-batch second-order graph, while preserving all transitions and the original loss/gradients up to floating-point accumulation. No finite differences or subsampling are used.

This trades extra recomputation and per-chunk overhead for lower EQM activation memory; it does not eliminate second derivatives or guarantee a speedup. Smaller chunks generally reduce extra activation memory and can be slower. The ordinary compressor/CE/MSE graphs still consume memory. Chunking is valid because this predictor/head process transitions independently; cross-token attention, batch-dependent layers, or stochastic predictor layers would require a different implementation.

EQM runs in FP32 with autocast disabled. AMP scales only the outer gradients, never the inner energy field. Memory remains connected to the compressor. Validation computes EQM using local action gradients inside the surrounding `no_grad` context, without retaining a parameter-update graph.

An isolated CPU benchmark (256 transitions, width 64, three predictor layers, action dimension 16, chunks of 32) produced the same loss for both implementations. Peak unique storage retained by autograd saves was 5.46 MiB for the full reference versus 1.14 MiB for recomputation (about 4.8x lower). Uninstrumented forward/backward time was about 44 ms versus 140 ms: recomputation was slower. This saved-storage proxy is not process RSS, full-training memory, or CUDA peak allocation.

Run the benchmark on a T4 to measure actual CUDA allocator peaks and timing:

```bash
python benchmark_eqm.py --device cuda --tokens 4096 --width 256 \
  --action-dim 32 --layers 3 --chunk-size 128
```

The benchmark isolates EQM; full training also retains the ordinary compressor/CE/MSE graphs. Tune `--eqm-chunk-size` against measured memory and throughput on your hardware.

## Run on two T4 GPUs

Use Python 3.10+ and a CUDA-enabled PyTorch installation appropriate for your machine.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prepare.py
wandb login
torchrun --standalone --nproc_per_node=2 train.py --run-name shakespeare-world-model
```

Dataset: [Karpathy's Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare), with a contiguous 90% training / 10% validation split and a character vocabulary. The download is explicit; training can also read an existing UTF-8 file with `--data`.

Defaults: 20,000 optimizer steps, context 256, width 256, four compressor layers, three predictor layers, 32-dimensional actions. Each GPU samples 16 independent random windows per microbatch, with four accumulation microsteps: effective batch 128 windows across two GPUs. DDP synchronizes gradients on the last microstep. CUDA uses FP16 autocast and gradient scaling, suitable for T4 GPUs; GPU memory and throughput need measurement on your hardware. Reduce `--batch-size` or `--context` if needed. CPU uses FP32. See [PyTorch DDP documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).

Every 500 optimizer steps (and at the final step), validation and planning run. Checkpoints are saved every 500 steps, on validation improvement, and at the end. Numbered checkpoint files retain history; `latest.pt` and `best.pt` point to the relevant files. Resume with the same model/data/training arguments and `--resume checkpoints/latest.pt`. Checkpoints restore model, AdamW, scaler, step, best validation loss, vocabulary, W&B ID, and per-rank Python/NumPy/Torch/CUDA RNG state. Keep world size unchanged for RNG continuity. Keep total `--steps` unchanged to preserve the learning-rate schedule. Only load trusted checkpoints.

W&B logs total loss, raw CE/MSE/KL/EQM, learning rate, gradient norm, validation losses, planning objective history, and a table of generated text. `--wandb-mode offline` records locally; `--wandb-mode disabled` needs no account. Planning results also go to `checkpoints/plan_STEP.json`.

## Planning semantics

The prompt is compressed once. A 500-token horizon uses **499 initially Gaussian actions and transitions**. The first next-token prediction comes from the prompt's last memory; each transition supplies the memory used to predict the subsequent token. Predicted memory becomes the next predictor input. The model is frozen and dropout disabled; only actions are optimized. Neither the compressor nor the action encoder runs inside the optimization loop.

The goal string (default `" ."`) is used **only as a cross-entropy target** for the final `len(goal)` positions of the 500-token rollout. No goal character or its embedding is fed into any module. All goal positions are reached using optimized actions and recursively predicted memories. Multi-character goals are scored jointly on that same free rollout, with no teacher forcing.

```
memory = compressor(prompt)[-1]  # once, detached
logits[0] = head(memory)
for t in range(horizon - 1):
    memory = predictor(memory, optimized_actions[t])
    logits[t + 1] = head(memory)
loss = CE(logits[-len(goal):], goal)
```

Planning uses only goal CE. There is no action regularizer and no `--plan-action-weight` option. The best objective candidate, including initialization and the final update, is retained. W&B and JSON contain the full decoded `rollout`, the actual decoded `predicted_goal` suffix, goal CE at every planning iteration, and `goal_exact_match`. Exact match checks **every** character of the suffix. Optimizing the loss cannot guarantee a match; the recorded match value reports whether it succeeded. No target text is appended to the generated output.

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --goal 'THE END.' --plan-horizon 500 --plan-iterations 100
```

The head is now trained as `head(m[t]) -> x[t+1]`; applying it to predicted `m[t+1]` predicts `x[t+2]`. Planning includes the prompt memory first to avoid skipping its next token. A one-token horizon uses no actions, so its prediction cannot be changed by optimizing future actions. Goal text is never a model input.

The horizon includes the goal: this example generates 500 tokens total, with the last eight positions optimized toward `THE END.`. Long differentiable rollouts can be expensive and can exploit model errors. A low goal loss does not guarantee coherent intermediate text or a realizable character-level trajectory. Predicted memories need not equal compressor memories for the decoded path. This baseline implements one-step training as requested; multi-step consistency would be a separate experiment. Rank zero plans while other ranks wait, with a four-hour collective timeout. Planning restores training mode, parameter gradient flags, and training RNG streams.

## Checks

```bash
python -m unittest discover -s tests -v
```

Tests cover causal attention, next-token alignment, detached targets, finite gradients, planning improvement without changing model parameters, and exclusion of the action encoder and goal tokens from planning inputs. A quick end-to-end run:

```bash
python train.py --device cpu --wandb-mode disabled --steps 2 \
  --width 16 --heads 2 --compressor-layers 1 --predictor-layers 1 \
  --action-dim 4 --context 16 --batch-size 2 --accumulation 1 \
  --eval-batches 1 --plan-horizon 4 --plan-iterations 2
```

Local CPU checks use PyTorch 2.2.2, NumPy 1.26.4, and W&B 0.18.7 on an Intel Mac. All 18 tests passed, including attached compressor CE gradients, memory-pair action encoding, detached encoder inputs, default EQM disabling, unchanged legacy EQM equivalence, and planning alignment. A two-process CPU/Gloo run completed training with gradient accumulation, validation, planning, and checkpoint saving using compressor CE and EQM disabled by default. CUDA FP16, peak GPU memory, and NCCL behavior remain unverified on this machine; requirements target PyTorch 2.4+ for the GPU run.
