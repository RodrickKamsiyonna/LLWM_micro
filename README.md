# Character-level latent-action language model

The compressor produces a causal sequence of memories `h[t]`. A Gaussian action encoder infers latent actions from detached adjacent compressor memories. A **causal Transformer predictor** consumes memory/action pairs and predicts the next memory at each position. Its attention uses RoPE and is limited to past and current positions.

## Training

```python
h = compressor(input_tokens)
mean, log_std = action_encoder(detach(h), detach(next_memory))
action = mean + exp(log_std) * noise
predicted_memory = causal_predictor(h, action)
head_input = predicted_memory if head_training == "joint" else predicted_memory.detach()
logits = vocabulary_head(head_input)
loss = ce_weight * CE(logits, next_tokens) + kl_penalty + mse_weight * memory_loss(predicted_memory, detach(next_memory))
```

Use `--head-training joint` (default) to let token CE update the head, predictor, action encoder and compressor, or `--head-training detached` to train the vocabulary head from detached predicted memories. The memory prediction and KL objectives remain active in either mode. No EQM loss or detached compressor-memory probe is computed.

Example:

```bash
python train.py --target-kl 0.07 --head-training joint
python train.py --target-kl 0.07 --head-training detached
```

The predictor is trained in parallel on teacher-forced compressor memories with a causal attention mask. Planning uses the full prompt compressor-memory prefix, then recursively feeds predicted memories and optimized actions to the predictor using a bounded differentiable KV cache. The prompt-prefix actions are zero placeholders because no inferred prompt actions are available to the planner.

The predictor architecture differs from the old MLP, so start a fresh training run. Existing checkpoint resume requires matching architecture and vocabulary.
