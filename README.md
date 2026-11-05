# LLWM_micro: recursive multi-horizon latent-action world model

The causal Transformer compressor produces memories `h[t]`. The Gaussian action encoder infers **one-step** actions from consecutive true compressor memories: `a[t] = E(h[t], h[t+1])`. For every valid starting position `t`, the same action-conditioned MLP predictor is recursively unrolled for up to `--prediction-horizon k` steps:

```python
predicted = h[t]
for j in range(1, k + 1):
    predicted = predictor(predicted, action[t + j - 1])
    memory_loss += smooth_l1(predicted, h[t + j].detach())
    logits = vocabulary_head(predicted.detach() if detached else predicted)
    token_ce += CE(logits, token[t + j])
```

The second prediction receives the **first predicted memory**, not the true compressor memory. Overlapping windows cover every valid starting position. The memory and token losses are normalized over all valid (start, horizon) pairs; the Gaussian KL is calculated **once per adjacent action** to avoid counting a reused action multiple times.

`--head-training detached` (default) allows CE to train only the vocabulary head. `joint` allows CE gradients to reach the predictor and relevant upstream components. The action posterior receives detached true compressor memories in both modes. Memory regression still trains the predictor through its entire recursive unroll.

## Training

```bash
python train.py --target-kl 0.07 --prediction-horizon 4 --head-training detached
python train.py --target-kl 0.07 --prediction-horizon 4 --head-training joint
```

DistributedDataParallel support is unchanged; use `torchrun --standalone --nproc_per_node=2 train.py ...` for two GPUs.

## Planning

The planner encodes the prompt to obtain its starting memory and optimizes a sequence of actions through a recursive MLP rollout using **only vocabulary-head cross-entropy** on the final goal-length tokens:

```
planning_loss = goal_token_ce
```

Optimized predicted memories are decoded by the vocabulary head; actions are never decoded directly. No goal-memory MSE, inverse-dynamics planning loss, EQM, or memory probe is used.

The multi-horizon objective differs from the previous direct-transition training objective; start a fresh run or deliberately reinitialize training for a clean comparison.
