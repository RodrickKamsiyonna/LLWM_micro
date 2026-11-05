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



def mpc_plan(model, prompt, goal, horizon=50, execute_steps=5, iterations=100,
             lr=0.05, seed=123):
    """Goal-only MPC: plan remaining horizon, execute a block, re-encode tokens."""
    if not 1 <= execute_steps <= horizon or not 0 < goal.numel() <= horizon:
        raise ValueError("invalid execution horizon or goal length")
    if prompt.ndim != 2 or prompt.size(1) < 1 or goal.ndim != 1 or iterations < 1:
        raise ValueError("invalid prompt, goal or iterations")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("MPC learning rate must be finite and positive")
    training = model.training
    flags = [p.requires_grad for p in model.parameters()]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        prefix = prompt.detach()
        generated, rounds = [], []
        warm = None
        for start in range(0, horizon, execute_steps):
            remaining = horizon - start
            count = min(execute_steps, remaining)
            with torch.no_grad():
                memories, cache = model.encode(prefix[:, -model.config.context:])
                initial = memories[:, -1].detach()
            if warm is None:
                generator = torch.Generator(device=initial.device).manual_seed(seed + start)
                init = torch.randn((prefix.size(0), remaining, model.config.action_dim),
                                   generator=generator, device=initial.device)
            else:
                init = warm[:, previous_count:].detach().clone()
            actions = torch.nn.Parameter(init)
            optimizer = torch.optim.Adam([actions], lr=lr)
            targets = goal[None].expand(prefix.size(0), -1)
            best_loss, best_actions, initial_loss = float("inf"), None, None
            for iteration in range(iterations + 1):
                with torch.set_grad_enabled(iteration < iterations):
                    states = rollout(model, initial, actions)
                    logits = model.decode_memory(states[:, -goal.numel():])
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                           targets.reshape(-1))
                value = float(loss.detach())
                if not math.isfinite(value):
                    raise RuntimeError("non-finite MPC objective")
                if initial_loss is None:
                    initial_loss = value
                if value < best_loss:
                    best_loss, best_actions = value, actions.detach().clone()
                if iteration < iterations:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([actions], 1.)
                    optimizer.step()
            with torch.no_grad():
                for j in range(count):
                    # Decode the executed action from the real compressor memory.
                    token = model.predict_logits(initial, best_actions[:, j]).argmax(-1)
                    prefix = torch.cat((prefix, token[:, None]), dim=1)
                    generated.append(token)
                    # Update actual memory and KV cache from the observed token.
                    new_memory, cache = model.encode(token[:, None], kv_cache=cache)
                    initial = new_memory[:, -1]
            warm, previous_count = best_actions, count
            rounds.append({"start": start, "horizon": remaining,
                           "executed": count, "initial_loss": initial_loss,
                           "goal_ce": best_loss})
        tokens = torch.stack(generated, dim=1)
        suffix = tokens[:, -goal.numel():]
        return {"tokens": tokens.cpu(), "goal_tokens": suffix.cpu(),
                "goal_exact_match": suffix.eq(goal[None]).all(-1).float().mean().item(),
                "initial_loss": rounds[0]["initial_loss"],
                "goal_ce": rounds[-1]["goal_ce"], "best_loss": rounds[-1]["goal_ce"],
                "round_history": rounds}
    finally:
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)
        model.train(training)


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
