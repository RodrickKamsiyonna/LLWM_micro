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
        self.embedding_output = nn.Linear(c.width, c.width)

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

    def posterior(self, memory, next_embedding, *, detach_inputs=True):
        # Callers may detach posterior inputs or train through the memory path.
        if detach_inputs:
            memory, next_embedding = memory.detach(), next_embedding.detach()
        inputs = torch.cat((memory, next_embedding), -1)
        mean, log_std = self.action_encoder(inputs).chunk(2, -1)
        return mean, log_std.clamp(-5, 2)

    def predict_features(self, memory, action):
        """Independent action-conditioned MLP transition, supports batched sequences."""
        if memory.shape[:-1] != action.shape[:-1] or memory.size(-1) != self.config.width or action.size(-1) != self.config.action_dim:
            raise ValueError('memory/action batch dimensions or feature widths mismatch')
        for block in self.predictor:
            memory = block(memory, action)
        return self.predictor_norm(memory)

    def predict_embedding(self, memory, action):
        """Predict the next token embedding from compressor memory and action."""
        return self.embedding_output(self.predict_features(memory, action))

    def decode_memory(self, embedding, *, detach=False):
        """Negative squared distances to vocabulary embeddings (no learned CE head)."""
        x = (embedding.detach() if detach else embedding).float()
        vocabulary = self.embedding.weight.float()
        return -(x.square().sum(-1, keepdim=True)
                 + vocabulary.square().sum(-1) - 2*x @ vocabulary.T)

    def predict_logits(self, memory, action):
        return self.decode_memory(self.predict_embedding(memory, action))

    def forward(self, idx, targets, *, target_kl, kl_lambda=1., mse_weight=1.):
        """Train independent next-token embedding predictions from real prefixes."""
        for name, value in [('target_kl', target_kl), ('kl_lambda', kl_lambda),
                            ('mse_weight', mse_weight)]:
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if idx.ndim != 2 or idx.shape != targets.shape or idx.size(1) < 1:
            raise ValueError('idx and targets must have matching nonempty [batch, sequence] shapes')
        valid = targets.ne(-1)
        teacher = self.embedding(targets.clamp(min=0)).detach()
        memory, _ = self.encode(idx)
        mean, log_std = self.posterior(memory, teacher, detach_inputs=False)
        action = mean + log_std.exp()*torch.randn_like(mean)
        predicted = self.predict_embedding(memory, action)
        count = valid.sum().clamp(min=1)
        error = (predicted.float()-teacher.float()).square().mean(-1)
        mse = (error*valid).sum()/count
        kl = 0.5*(mean.float().square() + (2*log_std.float()).exp()-1-2*log_std.float())
        kl_total = (kl.sum(-1)*valid).sum()/count
        excess = F.relu(kl_total-target_kl)
        penalty = kl_lambda*excess.square()
        return {'loss': mse_weight*mse+penalty, 'embedding_mse': mse,
                'kl': kl_total/self.config.action_dim, 'kl_total': kl_total,
                'kl_excess': excess, 'kl_penalty': penalty}
