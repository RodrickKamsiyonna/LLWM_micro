"""Exact equilibrium matching with recomputed, bounded-size predictor graphs."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def equilibrium_matching_loss(model, memory, action, targets, *, eqm_lambda=1., chunk_size=128):
    """Chunk valid transitions; detach actions, preserve the memory gradient path.

    Non-reentrant checkpointing recomputes each chunk's higher-order graph during
    backward instead of retaining all predictor activations across chunks.
    """
    differentiable = torch.is_grad_enabled()
    valid = targets.reshape(-1).ne(-1)
    h = memory.reshape(-1, memory.size(-1))[valid]
    actual = action.detach().reshape(-1, action.size(-1))[valid].float()
    labels = targets.reshape(-1)[valid]
    if not labels.numel():
        return memory.float().sum()*0
    size = chunk_size or labels.numel()
    # Sample once so changing chunk size doesn't change the random objective.
    gamma = torch.rand(actual.size(0), 1, device=actual.device)
    eps = torch.randn_like(actual)
    noisy = (gamma*actual + (1-gamma)*eps).detach()
    target_gradient = ((eps-actual)*eqm_lambda*(1-gamma)).detach()

    def chunk_loss(states, noisy_action, target, tokens):
        # Explicit FP32 keeps the unscaled action derivative stable under AMP.
        with torch.enable_grad(), torch.autocast(device_type=states.device.type, enabled=False):
            noisy_action = noisy_action.detach().requires_grad_(True)
            logits = model.predict_logits(states.float(), noisy_action)
            energy = F.cross_entropy(logits, tokens, reduction='sum')
            gradient, = torch.autograd.grad(energy, noisy_action, create_graph=differentiable)
            return (gradient-target).square().sum()

    total = memory.new_zeros((), dtype=torch.float32)
    for start in range(0, labels.numel(), size):
        stop = start+size
        inputs = (h[start:stop], noisy[start:stop], target_gradient[start:stop], labels[start:stop])
        if differentiable:
            value = checkpoint(chunk_loss, *inputs, use_reentrant=False)
        else:
            value = chunk_loss(*inputs).detach()
        total = total+value
    return total/actual.numel()
