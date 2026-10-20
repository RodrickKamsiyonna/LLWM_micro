from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .eqm import equilibrium_matching


@dataclass
class ModelConfig:
    vocab_size: int
    context: int = 256
    width: int = 256
    heads: int = 4
    compressor_layers: int = 4
    predictor_layers: int = 3
    action_dim: int = 32
    dropout: float = 0.1
    rope_base: float = 10000.


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
    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c.width, elementwise_affine=False)
        self.condition = nn.Linear(c.action_dim, 2*c.width)
        self.net = nn.Sequential(nn.Linear(c.width, 4*c.width), nn.GELU(), nn.Linear(4*c.width, c.width))

    def forward(self, memory, action):
        shift, scale = self.condition(action).chunk(2, -1)
        return memory + self.net(self.norm(memory)*(1+scale) + shift)


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
        self.action_encoder = nn.Sequential(nn.Linear(2*c.width, 2*c.width), nn.GELU(), nn.Linear(2*c.width, 2*c.action_dim))
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

    def posterior(self, memory, next_token):
        # Keep the previously requested stop-gradients at encoder inputs.
        inputs = torch.cat((memory.detach(), self.embedding(next_token.clamp(min=0)).detach()), -1)
        mean, log_std = self.action_encoder(inputs).chunk(2, -1)
        return mean, log_std.clamp(-5, 2)

    def predict_features(self, memory, action):
        for block in self.predictor:
            memory = block(memory, action)
        return self.predictor_norm(memory)

    def predict_logits(self, memory, action):
        logits = self.head(self.predict_features(memory, action)).float()
        return 15 * torch.tanh(logits/15)

    def forward(self, idx, targets, *, kl_weight=0.001, ce_weight=1., eqm_weight=0., eqm_lambda=1., eqm_chunk_size=128):
        if idx.shape != targets.shape:
            raise ValueError('idx and targets must have matching [batch, sequence] shapes')
        h, _ = self.encode(idx)
        mean, log_std = self.posterior(h, targets)
        std = log_std.exp()
        action = mean + std*torch.randn_like(mean)
        logits = self.predict_logits(h, action)
        valid = targets.ne(-1)
        count = valid.sum().clamp(min=1)
        ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-1, reduction='sum') / count
        # Mean over valid tokens AND action dimensions, matching the supplied VAE KL.
        per_dim_kl = 0.5*(mean.float().square() + (2*log_std.float()).exp() - 1 - 2*log_std.float())
        kl = (per_dim_kl*valid[..., None]).sum()/(count*self.config.action_dim)
        eqm = ce.new_zeros(())
        if eqm_weight != 0:
            eqm = equilibrium_matching(self, h, action, targets, eqm_lambda, eqm_chunk_size)
        return {'loss': ce_weight*ce + kl_weight*kl + eqm_weight*eqm, 'ce': ce, 'kl': kl, 'eqm': eqm}
