import unittest
import torch
from llwm import ModelConfig, WorldModel
from llwm.planning import rollout, mpc_plan


class EmbeddingRolloutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        self.model = WorldModel(ModelConfig(8, context=4, width=16, heads=2,
            compressor_layers=1, predictor_layers=1, action_dim=4, dropout=0.))
        self.idx = torch.tensor([[0, 1, 2]])
        self.targets = torch.tensor([[1, 2, 3]])

    def test_embedding_target_and_action_inputs(self):
        seen, predicted = [], []
        hook = self.model.action_encoder.register_forward_pre_hook(
            lambda module, args: seen.append(args[0]))
        out = self.model.embedding_output.register_forward_hook(
            lambda module, args, value: predicted.append(value))
        from unittest.mock import patch
        with patch.object(self.model, 'encode', wraps=self.model.encode) as encode:
            values = self.model(self.idx, self.targets, target_kl=1.)
        self.assertEqual(encode.call_count, 1)
        self.assertIs(encode.call_args.args[0], self.idx)
        self.assertEqual(encode.call_args.kwargs, {})
        self.assertEqual(len(predicted), 1)
        hook.remove(); out.remove()
        teacher = self.model.embedding(self.targets).detach()
        torch.testing.assert_close(seen[0][..., 16:], teacher)
        torch.testing.assert_close(values['embedding_mse'],
            (predicted[0]-teacher).square().mean())
        values['loss'].backward()
        for module in [self.model.compressor, self.model.action_encoder, self.model.embedding_output]:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))

    def test_rollout_uses_compressor_and_keeps_action_gradients(self):
        h, cache = self.model.encode(self.idx)
        actions = torch.randn(1, 3, 4, requires_grad=True)
        states, embeddings = rollout(self.model, h, actions, cache, return_embeddings=True)
        memory = h[:, -1]
        expected = []
        for t in range(3):
            prediction = self.model.predict_embedding(memory, actions[:, t])
            torch.testing.assert_close(prediction, embeddings[:, t])
            h, cache = self.model.encode(embeddings=prediction[:, None], kv_cache=cache)
            memory = h[:, -1]
            expected.append(memory)
        torch.testing.assert_close(states, torch.stack(expected, 1))
        gradient, = torch.autograd.grad(states[:, -1].square().sum(), actions)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient[:, 0].abs().sum().item(), 0)

    def test_masking_and_mpc(self):
        values = self.model(self.idx, torch.full_like(self.targets, -1), target_kl=0.)
        self.assertEqual(values['loss'].item(), 0.)
        values['loss'].backward()
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        result = mpc_plan(self.model, self.idx, torch.tensor([3, 4]),
                          horizon=5, execute_steps=2, iterations=3)
        self.assertEqual(result['tokens'].shape, (1, 5))
        self.assertEqual([r['horizon'] for r in result['round_history']], [5, 3, 1])
        self.assertTrue(all(r['goal_mse'] <= r['initial_loss'] for r in result['round_history']))
        for k, v in self.model.state_dict().items():
            torch.testing.assert_close(v, before[k])
        self.assertTrue(self.model.training)
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))
