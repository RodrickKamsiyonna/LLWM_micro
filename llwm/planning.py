"""Optimize actions through autoregressive predicted-memory transitions."""
import math
import torch
from torch.nn import functional as F


def rollout(model, memory, actions):
    """Recur through predicted memories; decode each transition's added token."""
    memories = []
    for action in actions.unbind(1):
        memory = model.predict_features(memory, action)
        memories.append(memory)
    return model.decode_memory(torch.stack(memories, 1))


def plan(model, prompt, goal, horizon=500, iterations=100, lr=0.05, seed=123):
    if goal.ndim != 1 or goal.numel() < 1 or horizon < goal.numel() or iterations < 1:
        raise ValueError('goal must be nonempty; horizon >= goal length; iterations >= 1')
    if prompt.ndim != 2 or prompt.size(1) < 1:
        raise ValueError('prompt must be [batch, nonempty sequence]')
    training = model.training
    flags = [p.requires_grad for p in model.parameters()]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        with torch.no_grad():
            states, _ = model.encode(prompt[:, -model.config.context:])
            initial = states[:, -1].detach()
        generator = torch.Generator(device=initial.device).manual_seed(seed)
        actions = torch.nn.Parameter(torch.randn((initial.size(0), horizon, model.config.action_dim), generator=generator, device=initial.device))
        targets = goal[None].expand(initial.size(0), -1)
        optimizer = torch.optim.Adam([actions], lr=lr)
        history, best_loss, best_actions = [], float('inf'), None
        for i in range(iterations+1):
            with torch.set_grad_enabled(i < iterations):
                logits = rollout(model, initial, actions)
                loss = F.cross_entropy(logits[:, -goal.numel():].flatten(0, 1), targets.flatten())
            value = float(loss.detach())
            if not math.isfinite(value):
                raise RuntimeError('non-finite planning objective')
            exact = logits[:, -goal.numel():].argmax(-1).eq(targets).all(-1).float().mean()
            history.append({'iteration': i, 'loss': value, 'goal_ce': value, 'goal_exact_match': float(exact)})
            if value < best_loss:
                best_loss, best_actions = value, actions.detach().clone()
            if i < iterations:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([actions], 1.)
                optimizer.step()
        with torch.no_grad():
            logits = rollout(model, initial, best_actions)
            tokens = logits.argmax(-1)
            suffix = tokens[:, -goal.numel():]
        return {'tokens': tokens.cpu(), 'goal_tokens': suffix.cpu(), 'goal_exact_match': suffix.eq(targets).all(-1).float().mean().item(),
                'history': history, 'best_loss': best_loss, 'goal_ce': best_loss}
    finally:
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)
        model.train(training)
