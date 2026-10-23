import unittest
import torch
from torch.nn import functional as F
from llwm import ModelConfig, WorldModel
from llwm.planning import plan, rollout


class WorldModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = WorldModel(ModelConfig(8, context=4, width=16, heads=2, compressor_layers=2, predictor_layers=1, action_dim=4, dropout=0.))
        tokens = torch.randint(8, (2, 5))
        self.idx, self.targets = tokens[:, :-1], tokens[:, 1:]

    def test_compressor_is_causal(self):
        changed = self.idx.clone()
        changed[:, 2:] = (changed[:, 2:]+1) % 8
        actual, _ = self.model.encode(changed)
        expected, _ = self.model.encode(self.idx)
        torch.testing.assert_close(actual[:, :2], expected[:, :2])

    def test_predictor_adds_current_memory_after_final_norm_with_attached_skip(self):
        memory = torch.randn(2, 4, 16, requires_grad=True)
        action = torch.randn(2, 4, 4, requires_grad=True)
        # A constant delta isolates the outer skip from internal block residuals.
        with torch.no_grad():
            self.model.predictor_norm.weight.zero_()
            self.model.predictor_norm.bias.fill_(0.25)
        predicted = self.model.predict_features(memory, action)
        torch.testing.assert_close(predicted, memory+0.25)
        gradient, = torch.autograd.grad(predicted.sum(), memory)
        torch.testing.assert_close(gradient, torch.ones_like(memory))
        with torch.no_grad():
            self.model.predictor_norm.bias.zero_()
        torch.testing.assert_close(self.model.predict_features(memory, action), memory)

    def test_ce_kl_alignment_and_all_training_gradients(self):
        state = torch.get_rng_state()
        values = self.model(self.idx, self.targets, target_kl=1.)
        torch.set_rng_state(state)
        memory, cache = self.model.encode(self.idx)
        with torch.no_grad():
            last, _ = self.model.encode(self.targets[:, -1:], kv_cache=cache)
            next_memory = torch.cat((memory[:, 1:].detach(), last), 1)
        mean, log_std = self.model.posterior(memory, next_memory)
        action = mean + log_std.exp()*torch.randn_like(mean)
        logits = self.model.predict_logits(memory, action)
        expected_ce = F.cross_entropy(logits.flatten(0, 1), self.targets.flatten())
        expected_kl = torch.distributions.kl_divergence(
            torch.distributions.Normal(mean, log_std.exp()),
            torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))).mean()
        torch.testing.assert_close(values['ce'], expected_ce)
        torch.testing.assert_close(values['kl'], expected_kl)
        self.assertEqual(set(values), {'loss', 'ce', 'kl', 'kl_total', 'kl_excess', 'kl_penalty', 'mse', 'eqm', 'eqm_penalty', 'probe_ce', 'probe_accuracy'})
        values['ce'].backward()
        for name, p in self.model.named_parameters():
            if name.startswith('memory_probe.'):
                self.assertIsNone(p.grad, name)
                continue
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(p.grad.abs().sum().item(), 0., name)

    def test_action_encoder_uses_memory_pairs_and_detaches_inputs(self):
        memory, _ = self.model.encode(self.idx)
        memory.retain_grad()
        next_memory = torch.randn_like(memory, requires_grad=True)
        seen = []
        hook = self.model.action_encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
        try:
            mean, log_std = self.model.posterior(memory, next_memory)
        finally:
            hook.remove()
        torch.testing.assert_close(seen[0], torch.cat((memory.detach(), next_memory.detach()), -1))
        self.assertFalse(seen[0].requires_grad)
        (mean.square().mean()+log_std.square().mean()).backward()
        self.assertIsNone(memory.grad)
        self.assertIsNone(next_memory.grad)
        for module in [self.model.embedding, self.model.compressor, self.model.compressor_norm]:
            for p in module.parameters():
                self.assertIsNone(p.grad)
        for p in self.model.action_encoder.parameters():
            self.assertIsNotNone(p.grad)

    def test_ignored_targets_and_loss_weights(self):
        targets = self.targets.clone()
        targets[:, -1] = -1
        values = self.model(self.idx, targets, ce_weight=0.3, target_kl=0., kl_lambda=0.2)
        torch.testing.assert_close(values['loss'], 0.3*values['ce']+0.2*values['kl_total'].square()+values['mse'])
        ignored = self.model(self.idx, torch.full_like(targets, -1), target_kl=1.)
        self.assertEqual(ignored['loss'].item(), 0.)
        ignored['loss'].backward()

    def test_cached_encoder_matches_full_sliding_attention_past_context(self):
        self.model.eval()
        tokens = torch.randint(8, (1, 11))
        expected, _ = self.model.encode(tokens)
        actual, cache = self.model.encode(tokens[:, :3])
        pieces = [actual]
        for i in range(3, tokens.size(1)):
            actual, cache = self.model.encode(tokens[:, i:i+1], kv_cache=cache)
            pieces.append(actual)
            for k, v in cache.layers:
                self.assertLessEqual(k.size(2), self.model.config.context-1)
                self.assertEqual(k.shape, v.shape)
        torch.testing.assert_close(torch.cat(pieces, 1), expected, atol=2e-6, rtol=2e-5)
        self.assertEqual(cache.position, 11)

    def test_cached_chunk_matches_full_attention_and_context_one(self):
        self.model.eval()
        tokens = torch.randint(8, (1, 10))
        expected, _ = self.model.encode(tokens)
        prefix, cache = self.model.encode(tokens[:, :3])
        suffix, cache = self.model.encode(tokens[:, 3:], kv_cache=cache)
        torch.testing.assert_close(torch.cat((prefix, suffix), 1), expected, atol=2e-6, rtol=2e-5)
        one = WorldModel(ModelConfig(8, context=1, width=16, heads=2, compressor_layers=1, predictor_layers=1, action_dim=4, dropout=0.)).eval()
        _, cache = one.encode(tokens[:, :1])
        actual, cache = one.encode(tokens[:, 1:2], kv_cache=cache)
        expected, _ = one.encode(tokens[:, :2])
        torch.testing.assert_close(actual[:, -1], expected[:, -1])
        self.assertEqual(cache.layers[0][0].size(2), 0)

    def test_memory_rollout_matches_transitions_and_keeps_early_action_gradients(self):
        self.model.eval()
        initial = self.model.encode(self.idx[:1])[0][:, -1].detach()
        actions = torch.randn(1, 6, 4, requires_grad=True)
        actual = rollout(self.model, initial, actions)
        memory, predictions = initial, []
        for action in actions.unbind(1):
            memory = self.model.predict_features(memory, action)
            predictions.append(self.model.decode_memory(memory))
        torch.testing.assert_close(actual, torch.stack(predictions, 1))
        gradient, = torch.autograd.grad(F.cross_entropy(actual[:, -1], torch.tensor([2])), actions)
        self.assertGreater(gradient[:, 0].abs().sum().item(), 1e-10)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_mse_targets_are_shifted_detached_compressor_memories(self):
        tokens = torch.randint(8, (2, 5))
        idx, targets = tokens[:, :-1], tokens[:, 1:]
        predicted = []
        original = self.model.predict_features
        def predict(memory, action):
            result = original(memory, action)
            predicted.append(result)
            return result
        self.model.predict_features = predict
        values = self.model(idx, targets, target_kl=1., mse_weight=0.7)
        with torch.no_grad():
            expected, _ = self.model.encode(tokens)
        torch.testing.assert_close(values['mse'], F.mse_loss(predicted[0], expected[:, 1:]))
        gradient, = torch.autograd.grad(values['mse'], predicted[0], retain_graph=True)
        torch.testing.assert_close(gradient, 2*(predicted[0]-expected[:, 1:])/predicted[0].numel())
        torch.testing.assert_close(values['loss'], values['ce']+values['kl_penalty']+0.7*values['mse'])
        values['mse'].backward()
        for module in [self.model.compressor, self.model.predictor, self.model.action_encoder]:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))
        self.assertIsNone(self.model.head.weight.grad)

    def test_planning_optimizes_actions_only_and_skips_goal_conditioning(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        seen = []
        original = self.model.encode
        def encode(idx=None, **kwargs):
            seen.append((idx, kwargs))
            return original(idx, **kwargs)
        self.model.encode = encode
        self.model.embedding.weight.requires_grad_(False)
        result = plan(self.model, self.idx[:1, :3], torch.tensor([1, 2]), horizon=5, iterations=5)
        self.assertLess(result['best_loss'], result['history'][0]['loss'])
        # Only prompt compression, regardless of horizon or optimization iterations.
        self.assertEqual(len(seen), 1)
        torch.testing.assert_close(seen[0][0], self.idx[:1, :3])
        for item in result['history']:
            self.assertAlmostEqual(item['loss'], item['goal_ce']+item['consistency_penalty'], places=5)
            self.assertEqual(item['inverse_consistency'], item['consistency_penalty'])
        self.assertEqual(result['tokens'].shape, (1, 5))
        torch.testing.assert_close(result['goal_tokens'], result['tokens'][:, -2:])
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.embedding.weight.requires_grad)
        self.assertTrue(self.model.head.weight.requires_grad)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_one_action_plan_needs_no_feedback(self):
        self.model.eval()
        result = plan(self.model, self.idx[:1, :2], torch.tensor([1]), horizon=1, iterations=2)
        self.assertLess(result['best_loss'], result['history'][0]['loss'])
        self.assertEqual(result['tokens'].shape, (1, 1))

    def test_invalid_encoding_inputs(self):
        with self.assertRaises(ValueError):
            self.model.encode()
        with self.assertRaises(ValueError):
            self.model.encode(self.idx, embeddings=self.model.embedding(self.idx))
        with self.assertRaises(ValueError):
            plan(self.model, self.idx, torch.tensor([1]), horizon=0)

    def test_autocast_preserves_attention_dtypes_and_budget_gradients(self):
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            values = self.model(self.idx, self.targets, target_kl=0., kl_lambda=0.1)
        (values['loss']+values['probe_ce']).backward()
        for name, p in self.model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)


if __name__ == '__main__':
    unittest.main()
