import io
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from llwm import ModelConfig, WorldModel
from llwm.planning import plan, rollout
from train import arguments


class InverseConsistencyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.model = WorldModel(ModelConfig(8, width=16, heads=2, compressor_layers=1,
                                           predictor_layers=1, action_dim=4, dropout=0.)).eval()

    def test_each_transition_mean_and_full_action_gradient_match_reference(self):
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        initial = torch.randn(2, 16)
        actions = torch.randn(2, 5, 4, requires_grad=True)
        logits, consistency = rollout(self.model, initial, actions, return_consistency=True)
        memory, means, predictions = initial, [], []
        for action in actions.unbind(1):
            next_memory = self.model.predict_features(memory, action)
            # Explicitly use the Gaussian mean, without averaging trajectories or sampling.
            mean, _ = self.model.action_encoder(torch.cat((memory, next_memory), -1)).chunk(2, -1)
            means.append(mean)
            predictions.append(self.model.decode_memory(next_memory))
            memory = next_memory
        expected = F.mse_loss(actions, torch.stack(means, 1))
        torch.testing.assert_close(logits, torch.stack(predictions, 1))
        torch.testing.assert_close(consistency, expected)
        actual_gradient, = torch.autograd.grad(consistency, actions, retain_graph=True)
        expected_gradient, = torch.autograd.grad(expected, actions)
        torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertTrue(torch.isfinite(actual_gradient).all())
        self.assertGreater(actual_gradient[:, 0].abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_planning_posterior_keeps_both_memory_inputs_attached(self):
        memory = torch.randn(2, 3, 16, requires_grad=True)
        next_memory = torch.randn_like(memory, requires_grad=True)
        mean, _ = self.model.posterior(memory, next_memory, detach_inputs=False)
        gradients = torch.autograd.grad(mean.square().mean(), (memory, next_memory))
        for gradient in gradients:
            self.assertGreater(gradient.abs().sum().item(), 0.)

    def test_zero_weight_skips_inverse_encoder_and_weighted_best_metrics(self):
        prompt, goal = torch.tensor([[1, 2]]), torch.tensor([3, 4])
        with patch.object(self.model, 'posterior', side_effect=AssertionError('disabled')):
            result = plan(self.model, prompt, goal, horizon=4, iterations=2, consistency_weight=0.)
        self.assertEqual(result['best_loss'], result['goal_ce'])
        self.assertEqual(result['inverse_consistency'], 0.)
        result = plan(self.model, prompt, goal, horizon=4, iterations=3, consistency_weight=0.3)
        self.assertLess(result['best_loss'], result['history'][0]['loss'])
        self.assertAlmostEqual(result['best_loss'], result['goal_ce']+result['consistency_penalty'], places=5)
        self.assertAlmostEqual(result['consistency_penalty'], 0.3*result['inverse_consistency'])
        best = min(result['history'], key=lambda item: item['loss'])
        self.assertEqual(result['goal_ce'], best['goal_ce'])
        self.assertEqual(result['inverse_consistency'], best['inverse_consistency'])
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_cli_weight_default_override_and_invalid_values(self):
        with patch('sys.argv', ['train.py', '--target-kl', '1']):
            self.assertEqual(arguments().plan_consistency_weight, 1.)
        with patch('sys.argv', ['train.py', '--target-kl', '1', '--plan-consistency-weight', '0.2']):
            self.assertEqual(arguments().plan_consistency_weight, 0.2)
        for value in ['-1', 'nan', 'inf']:
            with patch('sys.argv', ['train.py', '--target-kl', '1', '--plan-consistency-weight', value]), patch('sys.stderr', io.StringIO()):
                with self.assertRaises(SystemExit):
                    arguments()
            with self.assertRaises(ValueError):
                plan(self.model, torch.tensor([[1]]), torch.tensor([2]), consistency_weight=float(value))
