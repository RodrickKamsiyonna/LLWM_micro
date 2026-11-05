"""Optimize actions using only vocabulary-head goal-token cross-entropy."""
import math
import torch
from torch.nn import functional as F


def rollout(model, memory, actions):
    """Recursively predict memories from actions using the MLP transition."""
    if memory.ndim == 3:
        memory = memory[:, -1]
    if memory.ndim != 2 or actions.ndim != 3 or memory.size(0) != actions.size(0):
        raise ValueError('memory must be [batch,width]; actions [batch,horizon,action_dim]')
    states = []
    for action in actions.unbind(1):
        memory = model.predict_features(memory, action)
        states.append(memory)
    return torch.stack(states, dim=1)


def plan(model, prompt, goal, horizon=500, iterations=100, lr=0.05, seed=123):
    """Optimize actions so the vocabulary head predicts the requested goal suffix."""
    if goal.ndim != 1 or goal.numel() < 1 or horizon < goal.numel() or iterations < 1:
        raise ValueError('goal must be nonempty; horizon >= goal length; iterations >= 1')
    if prompt.ndim != 2 or prompt.size(1) < 1:
        raise ValueError('prompt must be [batch, nonempty sequence]')
    training = model.training
    flags = [p.requires_grad for p in model.parameters()]
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    try:
        with torch.no_grad():
            prefix, _ = model.encode(prompt[:, -model.config.context:])
            initial = prefix[:, -1].detach()
            goal_ids = goal[None].expand(prompt.size(0), -1)
        generator = torch.Generator(device=initial.device).manual_seed(seed)
        actions = torch.nn.Parameter(torch.randn(
            (initial.size(0), horizon, model.config.action_dim),
            generator=generator, device=initial.device))
        optimizer = torch.optim.Adam([actions], lr=lr)
        history, best_loss, best_actions = [], float('inf'), None
        for i in range(iterations+1):
            with torch.set_grad_enabled(i < iterations):
                states = rollout(model, initial, actions)
                logits = model.decode_memory(states[:, -goal.numel():])
                goal_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), goal_ids.reshape(-1))
                objective = goal_ce
            value = float(objective.detach())
            if not math.isfinite(value):
                raise RuntimeError('non-finite planning objective')
            history.append({'iteration': i, 'loss': value, 'goal_ce': value})
            if value < best_loss:
                best_loss, best_actions = value, actions.detach().clone()
                best_goal_ce = value
            if i < iterations:
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                torch.nn.utils.clip_grad_norm_([actions], 1.)
                optimizer.step()
        with torch.no_grad():
            states = rollout(model, initial, best_actions)
            logits = model.decode_memory(states)
            tokens = logits.argmax(-1)
            suffix = tokens[:, -goal.numel():]
            targets = goal[None].expand(initial.size(0), -1)
            exact = suffix.eq(targets).all(-1).float().mean().item()
        return {'tokens': tokens.cpu(), 'goal_tokens': suffix.cpu(),
                'goal_exact_match': exact, 'history': history,
                'best_loss': best_loss, 'goal_ce': best_goal_ce}
    finally:
        for param, flag in zip(model.parameters(), flags):
            param.requires_grad_(flag)
        model.train(training)
