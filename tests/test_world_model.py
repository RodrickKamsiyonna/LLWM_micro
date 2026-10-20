import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from llwm import ModelConfig, WorldModel
from llwm.planning import plan, rollout


class WorldModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = WorldModel(ModelConfig(8, context=4, width=16, heads=2, compressor_layers=2, predictor_layers=1, action_dim=4, dropout=0.))
        self.idx = torch.randint(8, (2, 4))
        self.targets = torch.randint(8, (2, 4))

    def test_compressor_is_causal(self):
        changed = self.idx.clone()
        changed[:, 2:] = (changed[:, 2:]+1) % 8
        actual, _ = self.model.encode(changed)
        expected, _ = self.model.encode(self.idx)
        torch.testing.assert_close(actual[:, :2], expected[:, :2])

    def test_ce_kl_alignment_and_all_training_gradients(self):
        state = torch.get_rng_state()
        values = self.model(self.idx, self.targets)
        torch.set_rng_state(state)
        memory, _ = self.model.encode(self.idx)
        mean, log_std = self.model.posterior(memory, self.targets)
        action = mean + log_std.exp()*torch.randn_like(mean)
        logits = self.model.predict_logits(memory, action)
        expected_ce = F.cross_entropy(logits.flatten(0, 1), self.targets.flatten())
        expected_kl = torch.distributions.kl_divergence(
            torch.distributions.Normal(mean, log_std.exp()),
            torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))).mean()
        torch.testing.assert_close(values['ce'], expected_ce)
        torch.testing.assert_close(values['kl'], expected_kl)
        self.assertEqual(set(values), {'loss', 'ce', 'kl', 'eqm'})
        values['ce'].backward()
        for name, p in self.model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(p.grad.abs().sum().item(), 0., name)

    def test_action_encoder_uses_token_embeddings_and_detaches_inputs(self):
        memory, _ = self.model.encode(self.idx)
        memory.retain_grad()
        seen = []
        hook = self.model.action_encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
        try:
            mean, log_std = self.model.posterior(memory, self.targets)
        finally:
            hook.remove()
        torch.testing.assert_close(seen[0], torch.cat((memory.detach(), self.model.embedding(self.targets).detach()), -1))
        self.assertFalse(seen[0].requires_grad)
        (mean.square().mean()+log_std.square().mean()).backward()
        self.assertIsNone(memory.grad)
        for module in [self.model.embedding, self.model.compressor, self.model.compressor_norm]:
            for p in module.parameters():
                self.assertIsNone(p.grad)
        for p in self.model.action_encoder.parameters():
            self.assertIsNotNone(p.grad)

    def test_ignored_targets_and_loss_weights(self):
        targets = self.targets.clone()
        targets[:, -1] = -1
        values = self.model(self.idx, targets, ce_weight=0.3, kl_weight=0.2, eqm_weight=0.4, eqm_chunk_size=3)
        torch.testing.assert_close(values['loss'], 0.3*values['ce']+0.2*values['kl']+0.4*values['eqm'])
        ignored = self.model(self.idx, torch.full_like(targets, -1))
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

    def test_soft_cached_rollout_matches_full_recomputation_and_gradients(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        prompt = self.idx[:1, :3]
        with torch.no_grad():
            memories, cache = self.model.encode(prompt)
            initial = memories[:, -1]
            prefix = self.model.embedding(prompt)
        saved = [(k.clone(), v.clone()) for k, v in cache.layers]
        actions = torch.randn(1, 6, 4, requires_grad=True)
        cached = rollout(self.model, initial, cache, actions, temperature=0.8)
        cached_grad, = torch.autograd.grad(F.cross_entropy(cached[:, -1], torch.tensor([2])), actions)
        reference_actions = actions.detach().clone().requires_grad_()
        memory, sequence, predictions = initial, prefix, []
        for i, action in enumerate(reference_actions.unbind(1)):
            logits = self.model.predict_logits(memory, action)
            predictions.append(logits)
            if i+1 < reference_actions.size(1):
                soft = (logits/0.8).softmax(-1) @ self.model.embedding.weight
                sequence = torch.cat((sequence, soft[:, None]), 1)
                states, _ = self.model.encode(embeddings=sequence)
                memory = states[:, -1]
        reference = torch.stack(predictions, 1)
        reference_grad, = torch.autograd.grad(F.cross_entropy(reference[:, -1], torch.tensor([2])), reference_actions)
        torch.testing.assert_close(cached, reference, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(cached_grad, reference_grad, atol=2e-6, rtol=2e-4)
        self.assertGreater(cached_grad[:, 0].abs().sum().item(), 1e-10)
        for (k, v), (old_k, old_v) in zip(cache.layers, saved):
            torch.testing.assert_close(k, old_k)
            torch.testing.assert_close(v, old_v)

    def test_planning_optimizes_actions_only_and_skips_goal_conditioning(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        def forbidden(*args):
            raise AssertionError('planning must not use posterior or see goal tokens')
        self.model.posterior = forbidden
        self.model.action_encoder.forward = forbidden
        seen = []
        original = self.model.encode
        def encode(idx=None, **kwargs):
            seen.append((idx, kwargs))
            return original(idx, **kwargs)
        self.model.encode = encode
        self.model.embedding.weight.requires_grad_(False)
        result = plan(self.model, self.idx[:1, :3], torch.tensor([1, 2]), horizon=5, iterations=5)
        self.assertLess(result['best_loss'], result['history'][0]['loss'])
        # One prompt prefill; six candidates plus final decode, four feedback steps each.
        self.assertEqual(len(seen), 1 + 7*4)
        torch.testing.assert_close(seen[0][0], self.idx[:1, :3])
        self.assertTrue(all(idx is None and 'embeddings' in kwargs for idx, kwargs in seen[1:]))
        for item in result['history']:
            self.assertEqual(item['loss'], item['goal_ce'])
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
            plan(self.model, self.idx, torch.tensor([1]), temperature=0)

    def test_autocast_preserves_attention_dtypes_and_eqm_gradients(self):
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            values = self.model(self.idx, self.targets, eqm_weight=0.1, eqm_chunk_size=3)
        values['loss'].backward()
        for name, p in self.model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)


if __name__ == '__main__':
    unittest.main()
