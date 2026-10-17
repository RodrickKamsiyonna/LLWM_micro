import unittest
import torch
from torch.nn import functional as F
from llwm import ModelConfig, WorldModel
from llwm.planning import plan, rollout


class WorldModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = WorldModel(ModelConfig(8, context=8, width=16, heads=2, compressor_layers=1, predictor_layers=1, action_dim=4, dropout=0.))
        self.tokens = torch.randint(8, (2, 7))

    def test_compressor_is_causal(self):
        changed = self.tokens.clone()
        changed[:, 4:] = (changed[:, 4:]+1) % 8
        torch.testing.assert_close(self.model.compress(changed)[:, :4], self.model.compress(self.tokens)[:, :4])

    def test_losses_align_and_target_is_detached(self):
        captured = {}
        original = self.model.compress
        def compress(tokens):
            result = original(tokens)
            result.retain_grad()
            captured['memory'] = result
            return result
        self.model.compress = compress
        state = torch.get_rng_state()
        values = self.model(self.tokens)
        memory = captured['memory']
        torch.set_rng_state(state)
        mu, log_std = self.model.posterior(memory[:, :-1], self.tokens[:, 1:])
        predicted = self.model.transition(memory[:, :-1], mu + log_std.exp()*torch.randn_like(mu))
        expected = F.cross_entropy(self.model.head(predicted).flatten(0, 1), self.tokens[:, 1:].flatten())
        torch.testing.assert_close(expected, values['ce'])
        torch.testing.assert_close(values['mse'], F.mse_loss(predicted, memory[:, 1:].detach()))
        expected_kl = torch.distributions.kl_divergence(
            torch.distributions.Normal(mu, log_std.exp()),
            torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))).sum(-1).mean()
        torch.testing.assert_close(values['kl'], expected_kl)
        values['loss'].backward()
        self.assertEqual(memory.grad[:, -1].abs().sum().item(), 0.)
        self.assertGreater(memory.grad[:, :-1].abs().sum().item(), 0.)
        for name, p in self.model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_planning_changes_actions_not_weights_and_restores_flags(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.embedding.weight.requires_grad_(False)
        result = plan(self.model, self.tokens[:1, :3], torch.tensor([1, 2]), horizon=4, iterations=8)
        self.assertLess(result['best_loss'], result['history'][0]['loss'])
        self.assertLess(min(h['goal_ce'] for h in result['history']), result['history'][0]['goal_ce'])
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.embedding.weight.requires_grad)
        self.assertTrue(self.model.head.weight.requires_grad)
        for k, v in self.model.state_dict().items():
            torch.testing.assert_close(v, before[k])
        self.assertEqual(result['tokens'].shape, (1, 4))
        self.assertEqual(result['goal_tokens'].shape, (1, 2))

    def test_terminal_goal_gradients_reach_first_action(self):
        memory = self.model.compress(self.tokens[:1, :3])[:, -1].detach()
        actions = torch.randn(1, 5, 4, requires_grad=True)
        terminal, _ = rollout(self.model, memory, actions)
        loss = F.cross_entropy(self.model.head(terminal), torch.tensor([2]))
        gradient, = torch.autograd.grad(loss, actions)
        self.assertGreater(gradient[:, 0].abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_planning_never_encodes_goal_tokens(self):
        def forbidden(*args):
            raise AssertionError('planning must not use the action encoder')
        self.model.posterior = forbidden
        self.model.action_encoder.forward = forbidden
        calls = []
        original = self.model.compress
        def compress(tokens):
            calls.append(tokens.clone())
            return original(tokens)
        self.model.compress = compress
        result = plan(self.model, self.tokens[:1, :3], torch.tensor([1, 2]), horizon=4, iterations=2)
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(calls[0], self.tokens[:1, :3])
        torch.testing.assert_close(result['goal_tokens'], result['tokens'][:, -2:])
        self.assertIn(result['goal_exact_match'], (0., 1.))


if __name__ == '__main__':
    unittest.main()
