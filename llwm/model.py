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


class CausalBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.heads
        self.ln1 = nn.LayerNorm(c.width)
        self.qkv = nn.Linear(c.width, 3 * c.width)
        self.out = nn.Linear(c.width, c.width)
        self.ln2 = nn.LayerNorm(c.width)
        self.mlp = nn.Sequential(nn.Linear(c.width, 4*c.width), nn.GELU(), nn.Linear(4*c.width, c.width), nn.Dropout(c.dropout))
        self.dropout = c.dropout

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = [z.view(b, t, self.heads, d//self.heads).transpose(1, 2) for z in self.qkv(self.ln1(x)).chunk(3, -1)]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.)
        x = x + self.out(a.transpose(1, 2).contiguous().view(b, t, d))
        return x + self.mlp(self.ln2(x))


class AdaptiveBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c.width, elementwise_affine=False)
        self.condition = nn.Linear(c.action_dim, 2*c.width)
        self.net = nn.Sequential(nn.Linear(c.width, 4*c.width), nn.GELU(), nn.Linear(4*c.width, c.width))

    def forward(self, m, a):
        shift, scale = self.condition(a).chunk(2, -1)
        return m + self.net(self.norm(m) * (1 + scale) + shift)


class WorldModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        if c.width % c.heads:
            raise ValueError('width must be divisible by heads')
        self.config = c
        self.embedding = nn.Embedding(c.vocab_size, c.width)
        self.position = nn.Embedding(c.context + 1, c.width)
        self.compressor = nn.Sequential(*[CausalBlock(c) for _ in range(c.compressor_layers)], nn.LayerNorm(c.width))
        self.action_encoder = nn.Sequential(nn.Linear(2*c.width, 2*c.width), nn.GELU(), nn.Linear(2*c.width, 2*c.action_dim))
        self.predictor = nn.ModuleList([AdaptiveBlock(c) for _ in range(c.predictor_layers)])
        self.predictor_norm = nn.LayerNorm(c.width)
        self.head = nn.Linear(c.width, c.vocab_size)

    def compress(self, tokens):
        if tokens.size(1) > self.config.context + 1:
            raise ValueError('sequence exceeds compressor context')
        return self.compressor(self.embedding(tokens) + self.position(torch.arange(tokens.size(1), device=tokens.device)))

    def posterior(self, memory, token):
        # Train the encoder without sending its gradients into either input source.
        inputs = torch.cat((memory.detach(), self.embedding(token).detach()), -1)
        mu, log_std = self.action_encoder(inputs).chunk(2, -1)
        return mu, log_std.clamp(-5, 2)

    def transition(self, memory, action):
        for block in self.predictor:
            memory = block(memory, action)
        return self.predictor_norm(memory)

    def forward(self, tokens, kl_weight=0.001, mse_weight=1., ce_weight=1., eqm_weight=1., eqm_lambda=1., eqm_chunk_size=128):
        memories = self.compress(tokens)
        current, target = memories[:, :-1], memories[:, 1:].detach()
        mu, log_std = self.posterior(current, tokens[:, 1:])
        action = mu + log_std.exp() * torch.randn_like(mu)
        predicted = self.transition(current, action)
        ce = F.cross_entropy(self.head(predicted).float().flatten(0, 1), tokens[:, 1:].flatten())
        mse = F.mse_loss(predicted.float(), target.float())
        kl = 0.5 * (mu.float().square() + (2*log_std.float()).exp() - 1 - 2*log_std.float()).sum(-1).mean()
        eqm = equilibrium_matching(self, current, action, tokens[:, 1:], eqm_lambda, eqm_chunk_size) if eqm_weight else ce.new_zeros(())
        return {'loss': ce_weight*ce + mse_weight*mse + kl_weight*kl + eqm_weight*eqm,
                'ce': ce, 'mse': mse, 'kl': kl, 'eqm': eqm}
