# LLWM_micro: action-decoded character world model

The causal Transformer **compressor** encodes a token prefix into a sequence of memories. A Gaussian action encoder infers actions from adjacent memories. The **predictor is an action-conditioned MLP**, independently applied to each (memory, action) transition. Its memory regression is supervised by the detached next compressor memory.

The vocabulary decoder maps **actions directly to next-token logits**. Use `--head-training detached` (default) to feed `action.detach()` into the head, so its cross-entropy gradients update **only the head**. Use `--head-training joint` to feed the attached action, so CE also updates the action encoder through the sampled action (but not the compressor through its detached posterior inputs). The main loss contains memory regression and the KL budget; the separate `head_loss` is added to the backward objective without any shared gradient path. The `--ce-weight` option still scales the head CE. There is no EQM loss or memory probe.

## Training

```bash
python train.py --target-kl 0.07 --memory-loss smooth_l1 --head-training detached
python train.py --target-kl 0.07 --memory-loss smooth_l1 --head-training joint
```

## Goal-memory planning

1. Encode the prompt with the compressor.
2. Encode the goal **after the prompt** using its compressor KV cache, obtaining a contextual target goal memory.
3. Optimize a horizon-length sequence of actions through the recurrent MLP predictor to minimize MSE between the final predicted memory and the target goal memory.
4. Decode each optimized action through the action-only vocabulary head to produce a token sequence.

Planning uses **no token cross-entropy** and **no inverse-dynamics penalty**. The goal is a final memory-state target; when the horizon exceeds the number of goal tokens, the optimizer is free to choose intermediate actions. Decoded tokens may differ from the requested goal even when the final memory error is low. Planning reports both goal-memory MSE and exact decoded suffix match for diagnosis.

The architecture is incompatible with earlier predictor checkpoints; start a fresh run.
