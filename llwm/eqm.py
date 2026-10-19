"""Exact, recomputed EQM for a predictor with independent token transitions."""
import torch
from torch.nn import functional as F


def chunk_loss(model, memory, action, target, gamma, noise, strength, create_graph):
    mixed = (gamma * action.detach() + (1 - gamma) * noise).requires_grad_(True)
    logits = model.head(model.transition(memory, mixed)).float()
    energy = F.cross_entropy(logits, target, ignore_index=-1, reduction='sum')
    gradient, = torch.autograd.grad(energy, mixed, create_graph=create_graph)
    desired = (noise - action.detach()) * strength * (1 - gamma)
    return (gradient - desired).square().mean()


class RecomputedEQM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, memory, action, target, gamma, noise, model, chunk_size, strength, *params):
        ctx.save_for_backward(memory, action.detach(), target, gamma, noise, *params)
        ctx.model, ctx.chunk_size, ctx.strength = model, chunk_size, strength
        total = memory.new_zeros((), dtype=torch.float32)
        n = action.numel() // action.size(-1)
        h, a, y = memory.reshape(n, -1), action.reshape(n, -1), target.reshape(n)
        g, e = gamma.reshape(n, 1), noise.reshape(n, -1)
        # No graph survives a chunk. The inner gradient is unscaled FP32.
        with torch.enable_grad(), torch.autocast(device_type=memory.device.type, enabled=False):
            for start in range(0, n, chunk_size):
                stop = min(start + chunk_size, n)
                loss = chunk_loss(model, h[start:stop].detach().float(), a[start:stop].float(),
                                  y[start:stop], g[start:stop].float(), e[start:stop].float(), strength, False)
                total += loss.detach() * ((stop - start) / n)
        return total

    @staticmethod
    def backward(ctx, grad_output):
        memory, action, target, gamma, noise, *params = ctx.saved_tensors
        n = action.numel() // action.size(-1)
        h, a, y = memory.reshape(n, -1), action.reshape(n, -1), target.reshape(n)
        g, e = gamma.reshape(n, 1), noise.reshape(n, -1)
        h_grad = torch.zeros_like(h) if ctx.needs_input_grad[0] else None
        p_grads = [None] * len(params)
        active = [(i, p) for i, p in enumerate(params) if p.requires_grad]
        with torch.enable_grad(), torch.autocast(device_type=memory.device.type, enabled=False):
            for start in range(0, n, ctx.chunk_size):
                stop = min(start + ctx.chunk_size, n)
                local_h = h[start:stop].detach().float().requires_grad_(True)
                loss = chunk_loss(ctx.model, local_h, a[start:stop].float(), y[start:stop],
                                  g[start:stop].float(), e[start:stop].float(), ctx.strength, True)
                inputs = [local_h] + [p for _, p in active]
                # Scale after differentiating: AMP must not scale the energy field itself.
                gradients = torch.autograd.grad(loss, inputs, allow_unused=True)
                scale = grad_output * ((stop - start) / n)
                if h_grad is not None and gradients[0] is not None:
                    h_grad[start:stop] = (gradients[0] * scale).to(h_grad.dtype)
                for (i, _), gradient in zip(active, gradients[1:]):
                    if gradient is not None:
                        value = gradient * scale
                        p_grads[i] = value if p_grads[i] is None else p_grads[i] + value
        return (h_grad.reshape_as(memory) if h_grad is not None else None,
                None, None, None, None, None, None, None, *p_grads)


def equilibrium_matching(model, memory, action, targets, strength=1., chunk_size=128):
    if chunk_size < 1:
        raise ValueError('EQM chunk size must be positive')
    gamma = torch.rand((*action.shape[:-1], 1), device=action.device, dtype=torch.float32)
    noise = torch.randn_like(action, dtype=torch.float32)
    params = tuple(model.predictor.parameters()) + tuple(model.predictor_norm.parameters()) + tuple(model.head.parameters())
    return RecomputedEQM.apply(memory, action.detach(), targets, gamma, noise, model, chunk_size, strength, *params)
