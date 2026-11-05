# LLWM_micro: direct multi-horizon latent-action world model

The causal Transformer compressor encodes token sequences into memories. The Gaussian action encoder infers a *direct* action between each source memory and each future target memory, up to `--prediction-horizon k`. The action-conditioned MLP predictor maps each **original source memory and inferred action** directly to that target. There is **no recursive rollout during multi-horizon training**.

For each offset `j=1,...,k` and valid source `h[t]`:

```python
action[t,j] = action_encoder(h[t].detach(), h[t+j].detach())
predicted[t,j] = predictor(h[t], action[t,j])
memory_loss += distance(predicted[t,j], h[t+j].detach())
token_logits[t,j] = vocabulary_head(predicted[t,j].detach() if detached else predicted[t,j])
token_ce += CE(token_logits[t,j], token[t+j])
```

Memory regression, token CE and action KL are averaged over **all valid (start, offset) pairs**. `--head-training detached` (default) isolates the vocabulary head's CE gradients; `joint` lets CE also train the predictor and action encoder. Posterior inputs are detached in both modes, but the predictor's source memory remains differentiable.

## Training

```bash
python train.py --target-kl 0.07 --prediction-horizon 4 --head-training detached
python train.py --target-kl 0.07 --prediction-horizon 4 --head-training joint
```

## Planning

The planner encodes the prompt to obtain its starting memory. It optimizes a sequence of actions through an autoregressive MLP rollout using **only vocabulary-head cross-entropy on the final goal-length tokens**:

```
planning_loss = goal_token_ce
```

Optimized predicted memories are decoded by the vocabulary head; **actions are never decoded directly**. No inverse-dynamics planning loss, EQM or memory probe is used.

Note: training uses direct multi-horizon transitions while planning still rolls the MLP forward to produce a token sequence. The mismatch is intentional but should be evaluated experimentally. Changing the vocabulary head input size invalidates earlier head checkpoints; start a fresh training run.
