# LLWM_micro: embedding world model

The causal Transformer compressor maps a token prefix to latent memory `h`.
The Gaussian action encoder receives `h` and the next token embedding and
produces an action with 128 dimensions by default. The action-conditioned
predictor maps memory and action to a predicted next-token embedding.

Training minimizes MSE against detached token embeddings, plus the squared
excess above the configured Gaussian KL budget. There is no cross-entropy
training head. Training encodes the real token sequence once with causal
attention, infers one action per position from its memory and true next-token
embedding, and predicts that embedding in a single step. Loss and KL are
averaged over valid tokens. Predicted embeddings are fed back through the
compressor only during planning.

```bash
python train.py --target-kl 0.07 --action-dim 128
```

Use `--mse-weight` to scale embedding regression. The training
`--prediction-horizon` option has been removed. Previous vocabulary-head
checkpoints are incompatible with this architecture.

MPC encodes the goal token sequence as a desired terminal compressor state.
It optimizes actions through embedding/compressor rollouts using terminal-state
MSE, executes `--plan-execute-steps` actions, and replans with the remaining
horizon. Predicted embeddings become tokens by nearest squared embedding
distance; executed tokens update the real compressor cache. The remaining
actions provide a warm start for the next round. Goal-state proximity does not
guarantee an exact goal-token suffix; exact match is reported separately.

```bash
python -m unittest discover -s tests -p test_embedding_rollout.py
```

Older tests for CE heads, memory regression, probes, and EQM describe prior
architectures and need migration before they can validate this model. The EQM
module is retained as legacy code and is not used by training or MPC.
