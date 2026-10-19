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
        values = self.model(self.tokens, eqm_chunk_size=3)
        memory = captured['memory']
        torch.set_rng_state(state)
        mu, log_std = self.model.posterior(memory[:, :-1], memory[:, 1:])
        predicted = self.model.transition(memory[:, :-1], mu + log_std.exp()*torch.randn_like(mu))
        expected_ce = F.cross_entropy(self.model.head(predicted).flatten(0, 1), self.tokens[:, 1:].flatten())
        torch.testing.assert_close(values['ce'], expected_ce)
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

    def test_ce_trains_only_head_and_loss_weights(self):
        state = torch.get_rng_state()
        values = self.model(self.tokens, kl_weight=0.2, mse_weight=0.3, ce_weight=0.7, eqm_weight=0.4)
        expected = 0.2*values['kl'] + 0.3*values['mse'] + 0.7*values['ce'] + 0.4*values['eqm']
        torch.testing.assert_close(values['loss'], expected)
        values['ce'].backward()
        for module in [self.model.head]:
            for p in module.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0.)
        for module in [self.model.embedding, self.model.position, self.model.compressor, self.model.action_encoder, self.model.predictor, self.model.predictor_norm]:
            for p in module.parameters():
                self.assertIsNone(p.grad)
        torch.set_rng_state(state)
        disabled = self.model(self.tokens, eqm_weight=0.)
        torch.testing.assert_close(disabled['loss'], disabled['ce'] + disabled['mse'] + 0.001*disabled['kl'])

    def test_training_head_is_applied_only_to_predicted_memory(self):
        memories, inputs = [], []
        compressor_hook = self.model.predictor_norm.register_forward_hook(lambda module, args, output: memories.append(output))
        head_hook = self.model.head.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
        try:
            values = self.model(self.tokens, eqm_weight=0.)
        finally:
            compressor_hook.remove()
            head_hook.remove()
        self.assertEqual(set(values), {'loss', 'ce', 'mse', 'kl', 'eqm'})
        self.assertEqual(len(inputs), 1)
        torch.testing.assert_close(inputs[0], memories[0])
        self.assertFalse(inputs[0].requires_grad)
        self.assertIsNone(inputs[0].grad_fn)
        values['mse'].backward()
        for p in self.model.head.parameters():
            self.assertIsNone(p.grad)
        for module in [self.model.predictor, self.model.action_encoder]:
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0.)

    def test_action_encoder_detaches_inputs_but_trains_its_weights(self):
        memories = self.model.compress(self.tokens)
        memory, next_memory = memories[:, :-1], memories[:, 1:]
        memory.retain_grad()
        next_memory.retain_grad()
        mu, log_std = self.model.posterior(memory, next_memory)
        (mu.square().mean() + log_std.square().mean()).backward()
        self.assertIsNone(memory.grad)
        self.assertIsNone(next_memory.grad)
        for module in [self.model.embedding, self.model.position, self.model.compressor]:
            for p in module.parameters():
                self.assertIsNone(p.grad)
        for p in self.model.action_encoder.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0.)

    def test_memory_gradient_comes_only_from_direct_predictor_input(self):
        memories = self.model.compress(self.tokens)
        memory = memories[:, :-1]
        mu, log_std = self.model.posterior(memory, memories[:, 1:])
        action = mu + log_std.exp() * torch.randn_like(mu)
        predicted = self.model.transition(memory, action)
        target = torch.randn_like(predicted)
        loss = F.mse_loss(predicted, target)
        actual, = torch.autograd.grad(loss, memory)
        direct_memory = memory.detach().requires_grad_()
        direct_predicted = self.model.transition(direct_memory, action.detach())
        direct_loss = F.mse_loss(direct_predicted, target)
        expected, = torch.autograd.grad(direct_loss, direct_memory)
        torch.testing.assert_close(actual, expected)
        self.assertGreater(actual.abs().sum().item(), 0.)

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

    def test_planning_decodes_after_each_transition(self):
        self.model.eval()
        prompt = self.tokens[:1, :3]
        initial = self.model.compress(prompt)[:, -1]
        seen, transitions, outputs = [], [], []
        head_hook = self.model.head.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
        original = self.model.transition
        def transition(memory, action):
            transitions.append(action)
            result = original(memory, action)
            outputs.append(result.detach().clone())
            return result
        self.model.transition = transition
        try:
            result = plan(self.model, prompt, torch.tensor([1, 2]), horizon=4, iterations=2)
        finally:
            head_hook.remove()
        # Three objective evaluations plus final decode, each using H actions.
        self.assertEqual(len(transitions), 4*4)
        torch.testing.assert_close(seen[-1], torch.stack(outputs[-4:], dim=1))
        self.assertEqual(result['tokens'].size(1), 4)

    def test_one_token_plan_can_optimize_its_action(self):
        self.model.eval()
        prompt = self.tokens[:1, :3]
        result = plan(self.model, prompt, torch.tensor([1]), horizon=1, iterations=2)
        self.assertLess(result['best_loss'], result['history'][0]['loss'])
        self.assertEqual(len(result['history']), 3)
        for h in result['history']:
            self.assertEqual(h['loss'], h['goal_ce'])
            self.assertNotIn('action_l2', h)

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
