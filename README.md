# LLM world model: character-level Shakespeare

Four modules: a causal Transformer compressor, Gaussian action encoder, action-conditioned memory predictor with adaptive LayerNorm, and a vocabulary head. The predictor processes each training transition independently; planning recursively uses its own predicted memories.

For characters `x[0:T+1]`, the compressor produces `m[0:T+1]`. Each `m[t]` has access only to `x[:t+1]`. Training uses:

```
q(a[t] | m[t], x[t+1]) = Normal(mu, exp(log_std))
a[t] = mu + exp(log_std) * epsilon
predicted_m[t+1] = predictor(m[t], a[t])
logits[t+1] = head(predicted_m[t+1])
loss = ce_weight * CE(logits[t+1], x[t+1])
     + compressor_ce_weight * CE(head(m[t]), x[t+1])
     + mse_weight * MSE(predicted_m[t+1], stop_gradient(m[t+1]))
     + kl_weight * KL(q(a[t]) || Normal(0, I))
```

The action encoder receives detached copies of both `m[t]` and the next-token embedding. Its weights still learn through the sampled action and KL loss, but this branch cannot update the compressor or shared embeddings. The original `m[t]` remains connected to the predictor, so the compressor and its embeddings learn through that direct memory path. The additional compressor CE also trains the compressor and shared embeddings directly through the shared LLM head. The next-memory MSE target is also detached.

KL sums action dimensions and averages transitions. MSE averages memory dimensions and transitions. CE averages transitions. The default weights are predictor CE=1, compressor CE=1, MSE=1, KL=0.001. Set `--compressor-ce-weight` to change the new weight, or to `0` to disable its contribution. W&B logs it separately as `train/compressor_ce` and `val/compressor_ce`; `train/ce` and `val/ce` retain the predictor-based CE. The action encoder must receive the **next** token: the current token is already represented in `m[t]`. Predicted next memory is compared to detached next memory, not current memory.

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

W&B logs total loss, raw predictor CE/compressor CE/MSE/KL, learning rate, gradient norm, validation losses, planning objective history, and a table of generated text. `--wandb-mode offline` records locally; `--wandb-mode disabled` needs no account. Planning results also go to `checkpoints/plan_STEP.json`.

## Planning semantics

The prompt is compressed once to the last causal memory. Initialize **500 actions total** from the unit Gaussian prior. Freeze all model weights and optimize only these actions through the entire recursive predictor rollout. Model dropout is disabled. Neither the compressor nor the action encoder runs inside the optimization loop. The action encoder is used only during training.

The goal string (default `" ."`) is used **only as a cross-entropy target** for the final `len(goal)` positions of the 500-token rollout. No goal character or its embedding is fed into any module. All goal positions are reached using optimized actions and recursively predicted memories. Multi-character goals are scored jointly on that same free rollout, with no teacher forcing.

```
memory = compressor(prompt)[-1]  # once, detached
for t in range(horizon):
    memory = predictor(memory, optimized_actions[t])
    logits[t] = head(memory)
loss = CE(logits[-len(goal):], goal) + 0.001 * mean(actions**2)
```

The action penalty discourages extreme actions outside the training prior; set `--plan-action-weight 0` to use only goal CE. The best objective candidate, including initialization and the final update, is retained. W&B and JSON contain the full decoded `rollout`, the actual decoded `predicted_goal` suffix, goal CE at every planning iteration, and `goal_exact_match`. Exact match checks **every** character of the suffix. Optimizing the loss cannot guarantee a match; the recorded match value reports whether it succeeded. No target text is appended to the generated output.

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --goal 'THE END.' --plan-horizon 500 --plan-iterations 100
```

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

Local verification used PyTorch 2.2.2, NumPy 1.26.4, and W&B 0.18.7 on an Intel Mac (newer PyTorch wheels are unavailable for that platform). All eight unit tests passed, including compressor CE alignment, weighting, and gradient routing. A two-process CPU/Gloo run completed training, validation, planning, and checkpoint saving. Offline W&B produced both nonempty tables. Resuming step 1 exactly reproduced step 2 model/optimizer/scaler/RNG state, validation loss, and planning JSON. A 500-action CPU planning smoke check completed with decreasing objective. CUDA FP16 and NCCL execution on T4 GPUs remain unverified here; the requirements target PyTorch 2.4+ for the GPU run.
