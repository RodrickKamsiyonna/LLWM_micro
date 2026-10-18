import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from llwm import ModelConfig, WorldModel
from llwm.planning import plan


def arguments():
    p = argparse.ArgumentParser(description='Character-level latent-action world model')
    p.add_argument('--data', default='data/input.txt')
    p.add_argument('--out', default='checkpoints')
    p.add_argument('--resume')
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch-size', type=int, default=16, help='per GPU')
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--context', type=int, default=256)
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--compressor-layers', type=int, default=4)
    p.add_argument('--predictor-layers', type=int, default=3)
    p.add_argument('--action-dim', type=int, default=32)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--kl-weight', type=float, default=0.001)
    p.add_argument('--mse-weight', type=float, default=1.)
    p.add_argument('--ce-weight', type=float, default=1.)
    p.add_argument('--compressor-ce-weight', type=float, default=1.)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--eval-batches', type=int, default=20)
    p.add_argument('--plan-horizon', type=int, default=500)
    p.add_argument('--plan-iterations', type=int, default=100)
    p.add_argument('--plan-lr', type=float, default=0.05)
    p.add_argument('--plan-action-weight', type=float, default=0.001)
    p.add_argument('--prompt', default='ROMEO:\n')
    p.add_argument('--goal', default=' .')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    p.add_argument('--project', default='llm-world-model')
    p.add_argument('--run-name')
    a = p.parse_args()
    for name in ['steps', 'batch_size', 'accumulation', 'context', 'eval_every', 'save_every', 'eval_batches', 'plan_horizon', 'plan_iterations']:
        if getattr(a, name) < 1:
            p.error(f'--{name.replace("_", "-")} must be positive')
    if not a.prompt or not a.goal:
        p.error('prompt and goal must be nonempty')
    return a


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda'])


def main():
    args = arguments()
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local = int(os.environ.get('LOCAL_RANK', 0))
    cuda = args.device.startswith('cuda')
    if cuda:
        device = torch.device(f'cuda:{local}' if world > 1 else args.device)
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)
    if world > 1:
        dist.init_process_group(backend='nccl' if cuda else 'gloo', timeout=timedelta(hours=4))
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    text = Path(args.data).read_text(encoding='utf-8')
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    def encode(s):
        try:
            return torch.tensor([stoi[ch] for ch in s], dtype=torch.long, device=device)
        except KeyError as e:
            raise ValueError(f'character {e.args[0]!r} absent from dataset') from e
    def decode(t):
        return ''.join(chars[i] for i in t.tolist())
    prompt, goal = encode(args.prompt)[None], encode(args.goal)
    if args.plan_horizon < goal.numel():
        raise ValueError('planning horizon must include the entire goal sequence')
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    split = int(len(data)*0.9)
    datasets = {'train': data[:split], 'val': data[split:]}
    if min(map(len, datasets.values())) < args.context + 1:
        raise ValueError('both data splits need at least context+1 characters')
    def batch(which):
        source = datasets[which]
        starts = torch.randint(len(source)-args.context, (args.batch_size,))
        return torch.stack([source[i:i+args.context+1] for i in starts]).to(device)
    config = ModelConfig(len(chars), args.context, args.width, args.heads, args.compressor_layers, args.predictor_layers, args.action_dim, args.dropout)
    raw = WorldModel(config).to(device)
    optimizer = torch.optim.AdamW(raw.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = (torch.amp.GradScaler('cuda', enabled=cuda) if hasattr(torch.amp, 'GradScaler')
              else torch.cuda.amp.GradScaler(enabled=cuda))
    start, best_val, run_id = 0, float('inf'), None
    if args.resume:
        # Checkpoints include Python/NumPy RNG state: load only trusted local files.
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        if ckpt['config'] != asdict(config) or ckpt['chars'] != chars:
            raise ValueError('checkpoint model configuration or vocabulary differs')
        import hashlib
        if ckpt['data_sha256'] != hashlib.sha256(text.encode()).hexdigest():
            raise ValueError('checkpoint dataset differs')
        raw.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        start, best_val, run_id = ckpt['step'], ckpt['best_val'], ckpt.get('wandb_id')
        if len(ckpt['rng']) == world:
            restore_rng(ckpt['rng'][rank])
        elif rank == 0:
            print('World size changed; RNG streams restart from the configured seeds.', flush=True)
    model = DDP(raw, device_ids=[device.index] if cuda else None) if world > 1 else raw
    out = Path(args.out)
    run, wandb = None, None
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        if args.wandb_mode != 'disabled':
            import wandb
            settings = wandb.Settings()
            # Older SDKs otherwise query server capabilities when logging offline tables.
            if args.wandb_mode == 'offline' and hasattr(settings, 'allow_offline_artifacts'):
                settings = wandb.Settings(allow_offline_artifacts=False)
            run = wandb.init(project=args.project, name=args.run_name, config=vars(args), mode=args.wandb_mode,
                             id=run_id, resume='allow' if run_id else None, settings=settings)
        print(f'{sum(p.numel() for p in raw.parameters()):,} parameters; effective batch {args.batch_size*args.accumulation*world}', flush=True)
    def reduced(values):
        v = torch.stack([values[k].detach().float() for k in ['loss', 'ce', 'compressor_ce', 'mse', 'kl']])
        if world > 1:
            dist.all_reduce(v)
            v /= world
        return dict(zip(['loss', 'ce', 'compressor_ce', 'mse', 'kl'], v.tolist()))
    def evaluate():
        raw.eval()
        totals = {k: torch.zeros((), device=device) for k in ['loss', 'ce', 'compressor_ce', 'mse', 'kl']}
        with torch.no_grad():
            for _ in range(args.eval_batches):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cuda):
                    values = raw(batch('val'), args.kl_weight, args.mse_weight, args.ce_weight, args.compressor_ce_weight)
                for k in totals:
                    totals[k] += values[k] / args.eval_batches
        raw.train()
        return reduced(totals)
    def save(step, is_best=False):
        states = [None]*world if rank == 0 else None
        if world > 1:
            dist.gather_object(rng_state(), states, dst=0)
        else:
            states = [rng_state()]
        if rank == 0:
            import hashlib
            payload = {'model': raw.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                       'step': step, 'config': asdict(config), 'args': vars(args), 'chars': chars, 'rng': states,
                       'best_val': best_val, 'wandb_id': run.id if run else None, 'data_sha256': hashlib.sha256(text.encode()).hexdigest()}
            target = out / f'step_{step:07d}.pt'
            temporary = out / 'checkpoint.tmp'
            torch.save(payload, temporary)
            temporary.replace(target)
            # Latest/best pointers avoid writing multiple copies of large checkpoints.
            for name in ['latest.pt'] + (['best.pt'] if is_best else []):
                pointer = out / name
                pending = out / (name + '.tmp')
                if pending.is_symlink():
                    pending.unlink()
                pending.symlink_to(target.name)
                pending.replace(pointer)
    step, skipped_updates = start + 1, 0
    while step <= args.steps:
        raw.train()
        lr = args.lr * min(1., step/max(1, args.warmup)) * (0.1 + 0.9*0.5*(1+math.cos(math.pi*max(0, step-args.warmup)/max(1, args.steps-args.warmup))))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        totals = {k: torch.zeros((), device=device) for k in ['loss', 'ce', 'compressor_ce', 'mse', 'kl']}
        for micro in range(args.accumulation):
            sync = model.no_sync() if world > 1 and micro < args.accumulation-1 else nullcontext()
            with sync:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cuda):
                    values = model(batch('train'), args.kl_weight, args.mse_weight, args.ce_weight, args.compressor_ce_weight)
                if not torch.isfinite(values['loss']):
                    raise RuntimeError(f'nonfinite loss at step {step}')
                scaler.scale(values['loss']/args.accumulation).backward()
            for k in totals:
                totals[k] += values[k].detach()/args.accumulation
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.)
        if not cuda and not torch.isfinite(grad_norm):
            raise RuntimeError(f'nonfinite gradient at step {step}')
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < old_scale
        if skipped:
            skipped_updates += 1
            if rank == 0:
                print(f'AMP overflow: retrying optimizer step {step} at scale {scaler.get_scale()}', flush=True)
            continue
        metrics = {f'train/{k}': v for k, v in reduced(totals).items()}
        metrics.update({'train/lr': lr, 'train/grad_norm': float(grad_norm), 'train/amp_skipped_updates': skipped_updates, 'step': step})
        improved = False
        if step % args.eval_every == 0 or step == args.steps:
            # Evaluation/planning must not change subsequent training RNG streams.
            state = rng_state()
            torch.manual_seed(args.seed+10000+rank)
            val = evaluate()
            metrics.update({f'val/{k}': v for k, v in val.items()})
            improved = val['loss'] < best_val
            best_val = min(best_val, val['loss'])
            if rank == 0:
                result = plan(raw, prompt, goal, args.plan_horizon, args.plan_iterations, args.plan_lr, args.plan_action_weight)
                row = {'step': step, 'prompt': args.prompt, 'goal': args.goal, 'rollout': decode(result['tokens'][0]),
                       'predicted_goal': decode(result['goal_tokens'][0]),
                       'initial_loss': result['history'][0]['loss'], 'best_loss': result['best_loss'], 'goal_ce': result['goal_ce'], 'history': result['history']}
                row['goal_exact_match'] = result['goal_exact_match']
                (out / f'plan_{step:07d}.json').write_text(json.dumps(row, indent=2))
                print(json.dumps({k: v for k, v in row.items() if k != 'history'}), flush=True)
                metrics.update({'plan/initial_loss': row['initial_loss'], 'plan/best_loss': row['best_loss'], 'plan/goal_ce': row['goal_ce'], 'plan/goal_exact_match': row['goal_exact_match']})
                if run:
                    metrics['plan/text'] = wandb.Table(columns=['step', 'prompt', 'goal', 'rollout', 'predicted_goal'], data=[[step, args.prompt, args.goal, row['rollout'], row['predicted_goal']]])
                    metrics['plan/optimization'] = wandb.Table(columns=list(result['history'][0]), data=[list(h.values()) for h in result['history']])
            if world > 1:
                dist.barrier()
            restore_rng(state)
        if step % args.save_every == 0 or step == args.steps or improved:
            save(step, improved)
        if rank == 0:
            if run:
                run.log(metrics, step=step)
            if step == 1 or step % 10 == 0:
                print(json.dumps({k: v for k, v in metrics.items() if isinstance(v, (float, int))}), flush=True)
        step += 1
    if run:
        run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
