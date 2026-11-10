"""Differentiable embedding rollouts and goal-state MPC."""
import math
import torch
from torch.nn import functional as F


def rollout(model, memory, actions, kv_cache=None, *, return_embeddings=False):
    if memory.ndim == 3:
        memory = memory[:, -1]
    if memory.ndim != 2 or actions.ndim != 3 or actions.size(1) < 1:
        raise ValueError('expected memory [batch,width] and nonempty actions [batch,horizon,action_dim]')
    states, embeddings = [], []
    for action in actions.unbind(1):
        embedding = model.predict_embedding(memory, action)
        h, kv_cache = model.encode(embeddings=embedding[:, None], kv_cache=kv_cache)
        memory = h[:, -1]
        states.append(memory)
        embeddings.append(embedding)
    states = torch.stack(states, 1)
    return (states, torch.stack(embeddings, 1)) if return_embeddings else states


def mpc_plan(model, prompt, goal, horizon=50, execute_steps=5, iterations=100,
             lr=0.05, seed=123):
    if prompt.ndim != 2 or prompt.size(1) < 1 or goal.ndim != 1 or goal.numel() < 1:
        raise ValueError('prompt and goal must be nonempty token sequences')
    if not 1 <= execute_steps <= horizon or goal.numel() > horizon or iterations < 1:
        raise ValueError('invalid execution horizon, goal length or iterations')
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('MPC learning rate must be finite and positive')
    training = model.training
    flags = [p.requires_grad for p in model.parameters()]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        prefix, generated, rounds, warm = prompt.detach(), [], [], None
        with torch.no_grad():
            # The desired terminal state is the compressor representation of the goal.
            goal_state = model.encode(goal[None].expand(prompt.size(0), -1))[0][:, -1]
        for start in range(0, horizon, execute_steps):
            remaining = horizon-start
            count = min(execute_steps, remaining)
            with torch.no_grad():
                h, cache = model.encode(prefix[:, -model.config.context:])
                initial = h[:, -1]
            if warm is None:
                generator = torch.Generator(device=initial.device).manual_seed(seed)
                init = torch.randn(prefix.size(0), remaining, model.config.action_dim,
                                   generator=generator, device=initial.device, dtype=initial.dtype)
            else:
                init = warm[:, previous_count:].clone()
            actions = torch.nn.Parameter(init)
            optimizer = torch.optim.Adam([actions], lr=lr)
            best_loss, best_actions, initial_loss = float('inf'), None, None
            for iteration in range(iterations+1):
                with torch.set_grad_enabled(iteration < iterations):
                    states = rollout(model, initial, actions, cache)
                    loss = F.mse_loss(states[:, -1].float(), goal_state.float())
                value = float(loss.detach())
                if not math.isfinite(value):
                    raise RuntimeError('non-finite MPC objective')
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
                memory = initial
                for j in range(count):
                    embedding = model.predict_embedding(memory, best_actions[:, j])
                    token = model.decode_memory(embedding).argmax(-1)
                    prefix = torch.cat((prefix, token[:, None]), 1)
                    generated.append(token)
                    h, cache = model.encode(token[:, None], kv_cache=cache)
                    memory = h[:, -1]
            warm, previous_count = best_actions, count
            rounds.append(dict(start=start, horizon=remaining, executed=count,
                               initial_loss=initial_loss, goal_mse=best_loss))
        tokens = torch.stack(generated, 1)
        suffix = tokens[:, -goal.numel():]
        return dict(tokens=tokens.cpu(), goal_tokens=suffix.cpu(),
                    goal_exact_match=suffix.eq(goal[None]).all(-1).float().mean().item(),
                    initial_loss=rounds[0]['initial_loss'], best_loss=rounds[-1]['goal_mse'],
                    goal_mse=rounds[-1]['goal_mse'], round_history=rounds)
    finally:
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)
        model.train(training)


def plan(model, prompt, goal, horizon=500, iterations=100, lr=0.05, seed=123):
    return mpc_plan(model, prompt, goal, horizon=horizon, execute_steps=horizon,
                    iterations=iterations, lr=lr, seed=seed)
