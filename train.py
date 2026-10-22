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
from prepare import default_data_path, prepare_dataset

LOSS_KEYS = ('loss', 'ce', 'kl', 'kl_total', 'kl_excess', 'kl_penalty', 'mse', 'eqm', 'eqm_penalty', 'probe_ce', 'probe_accuracy')


def arguments():
    p = argparse.ArgumentParser(description='Character-level latent-action world model')
    p.add_argument('--dataset', choices=['shakespeare', 'wikitext'], default='shakespeare', help='download/select a text corpus (default: shakespeare)')
    p.add_argument('--data', default=None, help='custom local UTF-8 corpus; overrides --dataset download')
    p.add_argument('--out', default='checkpoints/token_action')
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
    p.add_argument('--target-kl', type=float, required=True, help='total KL budget in nats per valid token (sum over action dimensions)')
    p.add_argument('--kl-lambda', type=float, default=1., help='coefficient of squared KL excess above the target')
    p.add_argument('--kl-steps', type=int, default=0, help='linear coefficient warmup in optimizer updates; 0 disables, 1 reaches full strength immediately')
    p.add_argument('--ce-weight', type=float, default=1.)
    p.add_argument('--mse-weight', type=float, default=1., help='next-memory MSE coefficient')
    p.add_argument('--eqm-weight', type=float, default=0., help='EQM loss coefficient; 0 skips computation')
    p.add_argument('--eqm-lambda', type=float, default=1., help='target action-gradient field scale')
    p.add_argument('--eqm-chunk-size', type=int, default=128, help='valid transitions per recomputed EQM chunk; 0 uses all')
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--eval-batches', type=int, default=20)
    p.add_argument('--plan-horizon', type=int, default=500)
    p.add_argument('--plan-iterations', type=int, default=100)
    p.add_argument('--plan-lr', type=float, default=0.05)
    p.add_argument('--plan-consistency-weight', type=float, default=1., help='inverse-dynamics action MSE coefficient; 0 disables computation')
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
    if not math.isfinite(a.target_kl) or a.target_kl < 0:
        p.error('--target-kl must be finite and nonnegative')
    if not math.isfinite(a.kl_lambda) or a.kl_lambda < 0:
        p.error('--kl-lambda must be finite and nonnegative')
    if a.kl_steps < 0:
        p.error('--kl-steps must be nonnegative')
    if not math.isfinite(a.mse_weight) or a.mse_weight < 0:
        p.error('--mse-weight must be finite and nonnegative')
    if not math.isfinite(a.plan_consistency_weight) or a.plan_consistency_weight < 0:
        p.error('--plan-consistency-weight must be finite and nonnegative')
    for name in ['eqm_weight', 'eqm_lambda']:
        if not math.isfinite(getattr(a, name)) or getattr(a, name) < 0:
            p.error(f'--{name.replace("_", "-")} must be finite and nonnegative')
    if a.eqm_chunk_size < 0:
        p.error('--eqm-chunk-size must be nonnegative')
    if not a.prompt or not a.goal:
        p.error('prompt and goal must be nonempty')
    return a


def kl_lambda_at_step(step, peak, ramp_steps):
    """For N>=2: update 1 uses zero, update N and later use peak."""
    if ramp_steps < 0:
        raise ValueError('KL ramp length must be nonnegative')
    if ramp_steps == 0:
        return peak
    if ramp_steps == 1:
        return peak if step >= 1 else 0.
    return peak * min(1., max(0., (step-1)/(ramp_steps-1)))


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda'])


def clip_training_gradients(model, max_norm=1.):
    main_parameters = [p for name, p in model.named_parameters() if not name.startswith('memory_probe.')]
    main_norm = torch.nn.utils.clip_grad_norm_(main_parameters, max_norm)
    probe_norm = torch.nn.utils.clip_grad_norm_(model.memory_probe.parameters(), max_norm)
    return main_norm, probe_norm


def load_training_weights(model, optimizer, checkpoint):
    """Allow pre-probe checkpoints, preserving every existing optimizer state."""
    weights = dict(checkpoint['model'])
    current_weights = model.state_dict()
    action_keys = {key for key in current_weights if key.startswith('action_encoder.')}
    saved_action_keys = {key for key in weights if key.startswith('action_encoder.')}
    if action_keys != saved_action_keys or any(weights[key].shape != current_weights[key].shape for key in action_keys):
        raise ValueError('checkpoint action encoder differs from the halving-width architecture; start a new run')
    probe_keys = {f'memory_probe.{key}' for key in model.memory_probe.state_dict()}
    fresh_probe = not probe_keys.intersection(weights)
    if fresh_probe:
        current = model.state_dict()
        weights.update({key: current[key] for key in probe_keys})
    model.load_state_dict(weights)  # All non-probe architecture mismatches remain errors.
    saved = checkpoint['optimizer']
    if fresh_probe:
        groups = [dict(group, params=list(group['params'])) for group in saved['param_groups']]
        current_groups = optimizer.state_dict()['param_groups']
        if len(groups) != 1 or len(current_groups) != 1 or len(groups[0]['params'])+len(probe_keys) != len(current_groups[0]['params']):
            raise ValueError('pre-probe optimizer layout differs from this model')
        next_id = max(groups[0]['params'], default=-1)+1
        groups[0]['params'].extend(range(next_id, next_id+len(probe_keys)))
        saved = dict(saved, param_groups=groups)
    optimizer.load_state_dict(saved)
    return fresh_probe


def main():
    args = arguments()
    loss_options = dict(target_kl=args.target_kl, kl_lambda=args.kl_lambda, ce_weight=args.ce_weight, mse_weight=args.mse_weight,
                        eqm_weight=args.eqm_weight, eqm_lambda=args.eqm_lambda, eqm_chunk_size=args.eqm_chunk_size)
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
    # Download once on rank zero. Other ranks wait until the file is complete.
    if args.data is None:
        if rank == 0:
            data_path = prepare_dataset(args.dataset)
            print(f'Dataset: {args.dataset} ({data_path})', flush=True)
        if world > 1:
            dist.barrier()
        args.data = str(default_data_path(args.dataset))
    text = Path(args.data).read_text(encoding='utf-8')
    if not text:
        raise ValueError('dataset is empty')
    if args.dataset == 'wikitext' and args.prompt == 'ROMEO:\\n':
        args.prompt = 'The ' if set('The ').issubset(set(text)) else text[:1]
    if args.dataset == 'wikitext' and args.goal == ' .' and not set(args.goal).issubset(set(text)):
        args.goal = text[:1]
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
        window = torch.stack([source[i:i+args.context+1] for i in starts]).to(device)
        return window[:, :-1], window[:, 1:]
    config = ModelConfig(len(chars), args.context, args.width, args.heads, args.compressor_layers, args.predictor_layers, args.action_dim, args.dropout)
    raw = WorldModel(config).to(device)
    optimizer = torch.optim.AdamW(raw.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = (torch.amp.GradScaler('cuda', enabled=cuda) if hasattr(torch.amp, 'GradScaler')
              else torch.cuda.amp.GradScaler(enabled=cuda))
    start, best_val, run_id = 0, float('inf'), None
    loss_signature = {'objective': 'memory_pair_action_eqm_v5', 'ce': args.ce_weight, 'mse': args.mse_weight,
                      'eqm_weight': args.eqm_weight, 'eqm_lambda': args.eqm_lambda,
                      'target_kl': args.target_kl, 'kl_lambda': args.kl_lambda, 'kl_steps': args.kl_steps}
    if args.resume:
        # Checkpoints include Python/NumPy RNG state: load only trusted local files.
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        if ckpt['config'] != asdict(config) or ckpt['chars'] != chars:
            raise ValueError('checkpoint architecture/configuration or vocabulary differs; old memory-MSE checkpoints need a fresh run')
        import hashlib
        if ckpt['data_sha256'] != hashlib.sha256(text.encode()).hexdigest():
            raise ValueError('checkpoint dataset differs')
        fresh_probe = load_training_weights(raw, optimizer, ckpt)
        if fresh_probe and rank == 0:
            print('Initialized detached memory probe; restored existing model and optimizer states.', flush=True)
        scaler.load_state_dict(ckpt['scaler'])
        start, best_val, run_id = ckpt['step'], ckpt['best_val'], ckpt.get('wandb_id')
        if ckpt.get('loss_signature') != loss_signature:
            best_val = float('inf')
            if rank == 0:
                print('Training objective changed; resetting best validation loss.', flush=True)
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
        v = torch.stack([values[k].detach().float() for k in LOSS_KEYS])
        if world > 1:
            dist.all_reduce(v)
            v /= world
        return dict(zip(LOSS_KEYS, v.tolist()))
    def evaluate():
        raw.eval()
        totals = {k: torch.zeros((), device=device) for k in LOSS_KEYS}
        with torch.no_grad():
            for _ in range(args.eval_batches):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cuda):
                    values = raw(*batch('val'), **loss_options)
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
                       'best_val': best_val, 'loss_signature': loss_signature,
                       'kl_lambda_effective': kl_lambda_at_step(step, args.kl_lambda, args.kl_steps),
                       'wandb_id': run.id if run else None, 'data_sha256': hashlib.sha256(text.encode()).hexdigest()}
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
        effective_kl_lambda = kl_lambda_at_step(step, args.kl_lambda, args.kl_steps)
        train_loss_options = dict(loss_options, kl_lambda=effective_kl_lambda)
        lr = args.lr * min(1., step/max(1, args.warmup)) * (0.1 + 0.9*0.5*(1+math.cos(math.pi*max(0, step-args.warmup)/max(1, args.steps-args.warmup))))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        totals = {k: torch.zeros((), device=device) for k in LOSS_KEYS}
        for micro in range(args.accumulation):
            sync = model.no_sync() if world > 1 and micro < args.accumulation-1 else nullcontext()
            with sync:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cuda):
                    values = model(*batch('train'), **train_loss_options)
                if not torch.isfinite(values['loss']) or not torch.isfinite(values['probe_ce']):
                    raise RuntimeError(f'nonfinite loss at step {step}')
                scaler.scale((values['loss']+values['probe_ce'])/args.accumulation).backward()
            for k in totals:
                totals[k] += values[k].detach()/args.accumulation
        scaler.unscale_(optimizer)
        grad_norm, probe_grad_norm = clip_training_gradients(raw)
        if not cuda and (not torch.isfinite(grad_norm) or not torch.isfinite(probe_grad_norm)):
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
        metrics.update({'train/lr': lr, 'train/kl_lambda': effective_kl_lambda, 'train/grad_norm': float(grad_norm), 'train/probe_grad_norm': float(probe_grad_norm), 'train/amp_skipped_updates': skipped_updates, 'step': step})
        improved = False
        if step % args.eval_every == 0 or step == args.steps:
            # Evaluation/planning must not change subsequent training RNG streams.
            state = rng_state()
            torch.manual_seed(args.seed+10000+rank)
            val = evaluate()
            metrics.update({f'val/{k}': v for k, v in val.items()})
            metrics['val/kl_lambda'] = args.kl_lambda
            improved = val['loss'] < best_val
            best_val = min(best_val, val['loss'])
            if rank == 0:
                result = plan(raw, prompt, goal, args.plan_horizon, args.plan_iterations, args.plan_lr, consistency_weight=args.plan_consistency_weight)
                row = {'step': step, 'prompt': args.prompt, 'goal': args.goal, 'rollout': decode(result['tokens'][0]),
                       'predicted_goal': decode(result['goal_tokens'][0]),
                       'initial_loss': result['history'][0]['loss'], 'best_loss': result['best_loss'], 'goal_ce': result['goal_ce'], 'history': result['history']}
                row['inverse_consistency'] = result['inverse_consistency']
                row['consistency_penalty'] = result['consistency_penalty']
                row['goal_exact_match'] = result['goal_exact_match']
                (out / f'plan_{step:07d}.json').write_text(json.dumps(row, indent=2))
                print(json.dumps({k: v for k, v in row.items() if k != 'history'}), flush=True)
                metrics.update({'plan/initial_loss': row['initial_loss'], 'plan/best_loss': row['best_loss'], 'plan/goal_ce': row['goal_ce'], 'plan/goal_exact_match': row['goal_exact_match']})
                metrics.update({'plan/inverse_consistency': row['inverse_consistency'], 'plan/consistency_penalty': row['consistency_penalty']})
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
