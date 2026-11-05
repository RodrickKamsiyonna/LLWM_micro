"""Optimize actions using contextual goal memories and predicted-token head loss."""
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


def plan(model, prompt, goal, horizon=500, iterations=100, lr=0.05, seed=123, token_weight=1.0):
    """Match the final predicted memory to compressor(prompt + goal), then decode actions.

    The target is contextual: the goal tokens are encoded after the prompt, using
    the compressor cache. This avoids comparing unrelated absolute-position memories.
    """
    if not math.isfinite(token_weight) or token_weight < 0:
        raise ValueError('token_weight must be finite and nonnegative')
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
            prefix, cache = model.encode(prompt[:, -model.config.context:])
            initial = prefix[:, -1].detach()
            # The goal representation must include the prompt as causal context.
            goal_ids = goal[None].expand(prompt.size(0), -1)
            goal_memories, _ = model.encode(goal_ids, kv_cache=cache)
            goal_state = goal_memories[:, -1].detach()
        generator = torch.Generator(device=initial.device).manual_seed(seed)
        actions = torch.nn.Parameter(torch.randn(
            (initial.size(0), horizon, model.config.action_dim),
            generator=generator, device=initial.device))
        optimizer = torch.optim.Adam([actions], lr=lr)
        history, best_loss, best_actions = [], float('inf'), None
        for i in range(iterations+1):
            with torch.set_grad_enabled(i < iterations):
                states = rollout(model, initial, actions)
                goal_mse = F.mse_loss(states[:, -1].float(), goal_state.float())
                logits = model.decode_memory(states[:, -goal.numel():])
                goal_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), goal_ids.reshape(-1))
                objective = goal_mse + token_weight*goal_ce
            value = float(objective.detach())
            if not math.isfinite(value):
                raise RuntimeError('non-finite planning objective')
            history.append({'iteration': i, 'loss': value, 'goal_memory_mse': float(goal_mse.detach()), 'goal_ce': float(goal_ce.detach())})
            if value < best_loss:
                best_loss, best_actions = value, actions.detach().clone()
                best_memory_mse, best_goal_ce = float(goal_mse.detach()), float(goal_ce.detach())
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
                'best_loss': best_loss, 'goal_memory_mse': best_memory_mse, 'goal_ce': best_goal_ce}
    finally:
        for param, flag in zip(model.parameters(), flags):
            param.requires_grad_(flag)
        model.train(training)
