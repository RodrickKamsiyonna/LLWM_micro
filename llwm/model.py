from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    context: int = 256
    width: int = 256
    heads: int = 4
    compressor_layers: int = 4
    predictor_layers: int = 3
    action_dim: int = 128
    dropout: float = 0.1
    rope_base: float = 10000.
    action_layers: int = 2


@dataclass(frozen=True)
class KVCache:
    # Tuples and out-of-place updates allow reuse across optimizer iterations.
    layers: tuple
    position: int


def rotary(x, cos, sin):
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even*cos - odd*sin, even*sin + odd*cos), dim=-1).flatten(-2)


class CausalBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.context, self.dropout = c.heads, c.context, c.dropout
        self.ln1 = nn.LayerNorm(c.width)
        self.qkv = nn.Linear(c.width, 3*c.width)
        self.out = nn.Linear(c.width, c.width)
        self.ln2 = nn.LayerNorm(c.width)
        self.mlp = nn.Sequential(nn.Linear(c.width, 4*c.width), nn.GELU(), nn.Linear(4*c.width, c.width), nn.Dropout(c.dropout))

    def forward(self, x, cos, sin, position, past=None):
        b, t, d = x.shape
        q, k, v = [z.view(b, t, self.heads, d//self.heads).transpose(1, 2) for z in self.qkv(self.ln1(x)).chunk(3, -1)]
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        past_len = 0 if past is None else past[0].size(2)
        if past is not None:
            k, v = torch.cat((past[0], k), 2), torch.cat((past[1], v), 2)
        mask, causal = None, False
        if past_len == 0 and t <= self.context:
            causal = True
        elif t != 1 or k.size(2) > self.context:
            queries = torch.arange(position, position+t, device=x.device)[:, None]
            keys = torch.arange(position-past_len, position+t, device=x.device)[None, :]
            mask = (keys <= queries) & (keys > queries-self.context)
        # For a single cached query all retained keys are in its causal past.
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal,
                                           dropout_p=self.dropout if self.training else 0.)
        x = x + self.out(a.transpose(1, 2).contiguous().view(b, t, d))
        x = x + self.mlp(self.ln2(x))
        keep = self.context-1
        cache = (k[:, :, -keep:], v[:, :, -keep:]) if keep else (k[:, :, :0], v[:, :, :0])
        return x, cache


class AdaptiveBlock(nn.Module):
    """Action-conditioned MLP memory transition."""
    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c.width, elementwise_affine=False)
        self.condition = nn.Linear(c.action_dim, 2*c.width)
        self.net = nn.Sequential(nn.Linear(c.width, 4*c.width), nn.GELU(), nn.Linear(4*c.width, c.width))

    def forward(self, memory, action):
        shift, scale = self.condition(action).chunk(2, -1)
        return memory + self.net(self.norm(memory)*(1+scale) + shift)


class ActionEncoder(nn.Module):
    """Configurable-depth MLP that narrows to an action-dimensional bottleneck."""
    def __init__(self, width, action_dim=128, num_layers=2):
        super().__init__()
        if action_dim < 1 or action_dim > 2*width:
            raise ValueError('action_dim must be positive and no larger than 2*width')
        if num_layers < 1:
            raise ValueError('action encoder layer count must be positive')
        layers = []
        current = 2*width
        for layer in range(num_layers):
            following = action_dim if layer == num_layers-1 else max(action_dim, current//2)
            layers.extend((nn.Linear(current, following), nn.GELU()))
            current = following
        self.trunk = nn.Sequential(*layers)
        self.mean = nn.Linear(action_dim, action_dim)
        self.log_std = nn.Linear(action_dim, action_dim)

    def forward(self, inputs):
        hidden = self.trunk(inputs)
        return torch.cat((self.mean(hidden), self.log_std(hidden)), dim=-1)


class WorldModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        if min(c.context, c.width, c.heads) < 1 or c.width % c.heads or (c.width//c.heads) % 2:
            raise ValueError('context must be positive; width/head count must give an even head dimension')
        if min(c.compressor_layers, c.predictor_layers, c.action_dim) < 1:
            raise ValueError('layer counts and action dimension must be positive')
        self.config = c
        self.embedding = nn.Embedding(c.vocab_size, c.width)
        self.compressor = nn.ModuleList([CausalBlock(c) for _ in range(c.compressor_layers)])
        self.compressor_norm = nn.LayerNorm(c.width)
        freq = 1 / (c.rope_base ** (torch.arange(0, c.width//c.heads, 2).float()/(c.width//c.heads)))
        self.register_buffer('rope_frequency', freq, persistent=False)
        self.action_encoder = ActionEncoder(c.width, c.action_dim, c.action_layers)
        self.predictor = nn.ModuleList([AdaptiveBlock(c) for _ in range(c.predictor_layers)])
        self.predictor_norm = nn.LayerNorm(c.width)
        self.head = nn.Linear(c.width, c.vocab_size)

    def encode(self, idx=None, *, embeddings=None, kv_cache=None):
        """Encode IDs OR differentiable embeddings; return memories and bounded KV cache."""
        if (idx is None) == (embeddings is None):
            raise ValueError('provide exactly one of idx or embeddings')
        x = self.embedding(idx) if embeddings is None else embeddings
        if x.ndim != 3 or x.size(1) < 1 or x.size(-1) != self.config.width:
            raise ValueError('embeddings must have shape [batch, nonempty sequence, width]')
        position = 0 if kv_cache is None else kv_cache.position
        if kv_cache is not None and len(kv_cache.layers) != len(self.compressor):
            raise ValueError('cache layer count differs from compressor')
        angles = torch.arange(position, position+x.size(1), device=x.device).float()[:, None] * self.rope_frequency
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        caches = []
        for i, block in enumerate(self.compressor):
            x, layer_cache = block(x, cos, sin, position, None if kv_cache is None else kv_cache.layers[i])
            caches.append(layer_cache)
        return self.compressor_norm(x), KVCache(tuple(caches), position+x.size(1))

    def posterior(self, memory, next_memory, *, detach_inputs=True):
        # Training stops encoder-input gradients; planning differentiates transitions.
        if detach_inputs:
            memory, next_memory = memory.detach(), next_memory.detach()
        inputs = torch.cat((memory, next_memory), -1)
        mean, log_std = self.action_encoder(inputs).chunk(2, -1)
        return mean, log_std.clamp(-5, 2)

    def predict_features(self, memory, action):
        """Independent action-conditioned MLP transition, supports batched sequences."""
        if memory.shape[:-1] != action.shape[:-1] or memory.size(-1) != self.config.width or action.size(-1) != self.config.action_dim:
            raise ValueError('memory/action batch dimensions or feature widths mismatch')
        for block in self.predictor:
            memory = block(memory, action)
        return self.predictor_norm(memory)

    def decode_memory(self, memory, *, detach=False):
        """Decode predicted memories; detach only when training the head independently."""
        logits = self.head(memory.detach() if detach else memory).float()
        return 15 * torch.tanh(logits/15)

    def predict_logits(self, memory, action):
        return self.decode_memory(self.predict_features(memory, action))

    def forward(self, idx, targets, *, target_kl, kl_lambda=1., ce_weight=1., mse_weight=1.,
                memory_loss='smooth_l1', head_training='detached', prediction_horizon=4):
        if memory_loss not in ('mse', 'smooth_l1'):
            raise ValueError('memory_loss must be mse or smooth_l1')
        if not math.isfinite(target_kl) or target_kl < 0:
            raise ValueError('target_kl must be finite and nonnegative')
        if not math.isfinite(kl_lambda) or kl_lambda < 0:
            raise ValueError('kl_lambda must be finite and nonnegative')
        if not math.isfinite(mse_weight) or mse_weight < 0:
            raise ValueError('mse_weight must be finite and nonnegative')
        if head_training not in ('joint', 'detached'):
            raise ValueError('head_training must be joint or detached')
        if not isinstance(prediction_horizon, int) or prediction_horizon < 1:
            raise ValueError('prediction_horizon must be a positive integer')
        if idx.shape != targets.shape:
            raise ValueError('idx and targets must have matching [batch, sequence] shapes')
        h, cache = self.encode(idx)
        # Include the last target token in the compressor sequence. All future
        # targets for a given starting memory come from this same sequence.
        with torch.no_grad():
            last, _ = self.encode(targets[:, -1:].clamp(min=0), kv_cache=cache)
        future = torch.cat((h[:, 1:].detach(), last), dim=1)
        # Position t + j is represented by future[:, t+j-1].
        valid = targets.ne(-1)
        count = h.new_zeros(())
        ce_sum, memory_sum, kl_sum = count, count, count
        memory_loss_fn = F.mse_loss if memory_loss == 'mse' else F.smooth_l1_loss
        for j in range(1, min(prediction_horizon, h.size(1))+1):
            n = h.size(1)-j+1
            source = h[:, :n]
            destination = future[:, j-1:j-1+n]
            mean, log_std = self.posterior(source, destination)
            action = mean + log_std.exp()*torch.randn_like(mean)
            predicted = self.predict_features(source, action)
            logits = self.decode_memory(predicted, detach=(head_training == 'detached'))
            target_ids = targets[:, j-1:j-1+n]
            mask = valid[:, j-1:j-1+n]
            count = count + mask.sum()
            ce_sum = ce_sum + F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1),
                                                ignore_index=-1, reduction='sum')
            per_memory = memory_loss_fn(predicted.float(), destination.float(), reduction='none').mean(-1)
            memory_sum = memory_sum + (per_memory*mask).sum()
            per_dim_kl = 0.5*(mean.float().square() + (2*log_std.float()).exp() - 1 - 2*log_std.float())
            kl_sum = kl_sum + (per_dim_kl*mask[..., None]).sum()
        count = count.clamp(min=1)
        ce = ce_sum/count
        memory_error = memory_sum/count
        kl = kl_sum/(count*self.config.action_dim)
        kl_total = kl*self.config.action_dim
        kl_excess = F.relu(kl_total-target_kl)
        kl_penalty = kl_lambda*kl_excess.square()
        return {'loss': kl_penalty + mse_weight*memory_error, 'ce': ce, 'head_loss': ce_weight*ce, 'memory_loss': memory_error, 'kl': kl,
                'kl_total': kl_total, 'kl_excess': kl_excess, 'kl_penalty': kl_penalty}
