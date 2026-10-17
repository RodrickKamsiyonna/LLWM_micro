"""Goal tokens are loss targets only. Planning never invokes the action encoder."""
import math
import torch
from torch.nn import functional as F


def rollout(model, memory, actions):
    states = []
    for action in actions.unbind(1):
        memory = model.transition(memory, action)
        states.append(memory)
    return memory, torch.stack(states, 1)


def plan(model, prompt, goal, horizon=500, iterations=100, lr=0.05, action_weight=0.001, seed=123):
    """horizon counts ALL generated tokens; goal occupies the final len(goal) positions."""
    if goal.ndim != 1 or goal.numel() < 1 or horizon < goal.numel() or iterations < 1:
        raise ValueError('goal must be a nonempty vector; horizon >= goal length; iterations >= 1')
    if prompt.ndim != 2 or prompt.size(1) < 1:
        raise ValueError('prompt must have shape [batch, nonempty sequence]')
    training = model.training
    flags = [p.requires_grad for p in model.parameters()]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        with torch.no_grad():
            initial = model.compress(prompt[:, -model.config.context:])[:, -1].detach()
        generator = torch.Generator(device=initial.device).manual_seed(seed)
        actions = torch.nn.Parameter(torch.randn((initial.size(0), horizon, model.config.action_dim), generator=generator, device=initial.device))
        targets = goal[None].expand(initial.size(0), -1)
        optimizer = torch.optim.Adam([actions], lr=lr)
        history, best_value, best_actions = [], float('inf'), None
        # Include the initialization and final updated actions in candidate selection.
        for i in range(iterations + 1):
            _, states = rollout(model, initial, actions)
            logits = model.head(states[:, -goal.numel():]).float()
            ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            regularizer = actions.square().mean()
            loss = ce + action_weight * regularizer
            value = float(loss.detach())
            if not math.isfinite(value):
                raise RuntimeError('non-finite planning objective')
            exact = logits.argmax(-1).eq(targets).all(-1).float().mean()
            history.append({'iteration': i, 'loss': value, 'goal_ce': float(ce.detach()), 'action_l2': float(regularizer.detach()), 'goal_exact_match': float(exact)})
            if value < best_value:
                best_value, best_actions = value, actions.detach().clone()
            if i < iterations:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([actions], 1.)
                optimizer.step()
        with torch.no_grad():
            _, states = rollout(model, initial, best_actions)
            tokens = model.head(states).argmax(-1)
            suffix = tokens[:, -goal.numel():]
        return {'tokens': tokens.cpu(), 'goal_tokens': suffix.cpu(), 'goal_exact_match': suffix.eq(targets).all(-1).float().mean().item(),
                'history': history, 'best_loss': best_value, 'goal_ce': min(history, key=lambda h: h['loss'])['goal_ce']}
    finally:
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)
        model.train(training)
