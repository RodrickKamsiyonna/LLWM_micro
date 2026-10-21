"""Compare full-graph and exact recomputed EQM. CUDA reports real allocator peaks.

CPU reports unique storage bytes retained by autograd saves (a proxy, not RSS).
"""
import argparse
import gc
import json
import time
from contextlib import nullcontext
import torch
from llwm import ModelConfig, WorldModel
from llwm.eqm import chunk_loss, equilibrium_matching_loss


class SavedStorage:
    def __init__(self):
        self.refs, self.live, self.peak = {}, 0, 0

    def pack(self, tensor):
        tracker = self
        storage = tensor.untyped_storage()
        key, size = storage.data_ptr(), storage.nbytes()
        saved = tensor.detach()

        class Held:
            def __init__(self):
                self.tensor = saved
                count = tracker.refs.get(key, 0)
                if not count:
                    tracker.live += size
                tracker.refs[key] = count + 1
                tracker.peak = max(tracker.peak, tracker.live)

            def __del__(self):
                count = tracker.refs[key] - 1
                if count:
                    tracker.refs[key] = count
                else:
                    del tracker.refs[key]
                    tracker.live -= size

        return Held()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--tokens', type=int, default=256)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--action-dim', type=int, default=16)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--chunk-size', type=int, default=32)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(1)
    device = torch.device(args.device)
    model = WorldModel(ModelConfig(65, width=args.width, predictor_layers=args.layers, action_dim=args.action_dim)).to(device)
    memory = torch.randn(1, args.tokens, args.width, device=device, requires_grad=True)
    action = torch.randn(1, args.tokens, args.action_dim, device=device)
    target = torch.randint(65, (1, args.tokens), device=device)

    def run(kind, measure=False):
        model.zero_grad(set_to_none=True)
        memory.grad = None
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        tracker = SavedStorage()
        hooks = torch.autograd.graph.saved_tensors_hooks(tracker.pack, lambda held: held.tensor) if device.type == 'cpu' and measure else nullcontext()
        torch.manual_seed(2)
        started = time.perf_counter()
        with hooks:
            if kind == 'full':
                gamma = torch.rand(1, args.tokens, 1, device=device)
                noise = torch.randn_like(action)
                loss = chunk_loss(model, memory.reshape(-1, args.width), action.reshape(-1, args.action_dim),
                                  target.reshape(-1), gamma.reshape(-1, 1), noise.reshape(-1, args.action_dim), 1., True)
            else:
                loss = equilibrium_matching_loss(model, memory, action, target, chunk_size=args.chunk_size)
            loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return {'method': kind, 'loss': loss.item(), 'seconds': time.perf_counter()-started,
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                'peak_saved_storage_bytes_proxy': tracker.peak if device.type == 'cpu' else None}

    for kind in ['full', 'recomputed']:
        run(kind)  # warm up before timing
    for kind in ['full', 'recomputed']:
        timed = run(kind)
        measured = run(kind, measure=True)
        for key in ['peak_cuda_allocated_bytes', 'peak_saved_storage_bytes_proxy']:
            timed[key] = measured[key]
        print(json.dumps(timed), flush=True)


if __name__ == '__main__':
    main()
