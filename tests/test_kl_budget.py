import io
import unittest
from unittest.mock import patch

import torch
from llwm import ModelConfig, WorldModel


class KLBudgetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = WorldModel(ModelConfig(8, context=4, width=16, heads=2, compressor_layers=1, predictor_layers=1, action_dim=4, dropout=0.))
        self.idx = torch.randint(8, (2, 4))
        self.targets = torch.randint(8, (2, 4))

    def controlled(self, target, strength=3., targets=None):
        # N(1,1) versus N(0,1): 0.5 nats/dimension, 2 nats/token.
        mean = torch.ones(4, requires_grad=True)
        expanded = mean.expand(2, 4, 4)
        with patch.object(self.model, 'posterior', return_value=(expanded, torch.zeros_like(expanded))):
            values = self.model(self.idx, self.targets if targets is None else targets,
                                target_kl=target, kl_lambda=strength, ce_weight=0.7)
        return values, mean

    def test_total_is_dimension_sum_not_batch_sum(self):
        values, _ = self.controlled(1.)
        self.assertEqual(values['kl'].item(), 0.5)
        self.assertEqual(values['kl_total'].item(), 2.)
        self.assertEqual(values['kl_excess'].item(), 1.)
        self.assertEqual(values['kl_penalty'].item(), 3.)
        torch.testing.assert_close(values['loss'], 0.7*values['ce']+3.)

    def test_above_budget_gradient_matches_squared_excess(self):
        values, mean = self.controlled(1.)
        values['kl_penalty'].backward()
        torch.testing.assert_close(mean.grad, torch.full_like(mean, 6.))

    def test_below_and_at_budget_have_no_penalty_or_upward_push(self):
        for target in [2., 3.]:
            with self.subTest(target=target):
                values, mean = self.controlled(target)
                self.assertEqual(values['kl_penalty'].item(), 0.)
                torch.testing.assert_close(values['loss'], 0.7*values['ce'])
                values['kl_penalty'].backward()
                torch.testing.assert_close(mean.grad, torch.zeros_like(mean))

    def test_zero_coefficient_disables_penalty(self):
        values, mean = self.controlled(0., strength=0.)
        self.assertEqual(values['kl_total'].item(), 2.)
        self.assertEqual(values['kl_penalty'].item(), 0.)
        values['kl_penalty'].backward()
        torch.testing.assert_close(mean.grad, torch.zeros_like(mean))

    def test_ignored_positions_do_not_consume_budget(self):
        targets = self.targets.clone()
        targets[:, -1] = -1
        mean = torch.ones(2, 4, 4)
        mean[:, -1] = 99.
        mean.requires_grad_()
        with patch.object(self.model, 'posterior', return_value=(mean, torch.zeros_like(mean))):
            values = self.model(self.idx, targets, target_kl=1.)
        self.assertEqual(values['kl_total'].item(), 2.)
        values['kl_penalty'].backward()
        self.assertEqual(mean.grad[:, -1].abs().sum().item(), 0.)
        self.assertGreater(mean.grad[:, :-1].abs().sum().item(), 0.)
        ignored, _ = self.controlled(1., targets=torch.full_like(targets, -1))
        for value in ignored.values():
            self.assertEqual(value.item(), 0.)

    def test_target_is_required_and_coefficient_configurable(self):
        from train import arguments
        with patch('sys.argv', ['train.py', '--target-kl', '2.5', '--kl-lambda', '0.3']):
            args = arguments()
        self.assertEqual(args.target_kl, 2.5)
        self.assertEqual(args.kl_lambda, 0.3)
        with patch('sys.argv', ['train.py']), patch('sys.stderr', io.StringIO()):
            with self.assertRaises(SystemExit):
                arguments()

    def test_invalid_budget_and_coefficient_are_rejected(self):
        from train import arguments
        for option, value in [('--target-kl', '-1'), ('--target-kl', 'nan'), ('--target-kl', 'inf'),
                              ('--kl-lambda', '-1'), ('--kl-lambda', 'nan')]:
            with self.subTest(option=option, value=value):
                with patch('sys.argv', ['train.py', '--target-kl', '1', option, value]), patch('sys.stderr', io.StringIO()):
                    with self.assertRaises(SystemExit):
                        arguments()
        for target, strength in [(-1., 1.), (float('nan'), 1.), (1., -1.)]:
            with self.assertRaises(ValueError):
                self.model(self.idx, self.targets, target_kl=target, kl_lambda=strength)


if __name__ == '__main__':
    unittest.main()
