import io
import unittest
from unittest.mock import patch

import torch
from llwm import ModelConfig, WorldModel
from train import arguments, kl_lambda_at_step


class KLScheduleTests(unittest.TestCase):
    def test_linear_zero_start_exact_endpoint_and_plateau(self):
        expected = [0., 0., 0.5, 1., 1.5, 2., 2., 2.]
        actual = [kl_lambda_at_step(step, 2., 5) for step in range(8)]
        self.assertEqual(actual, expected)
        self.assertEqual(kl_lambda_at_step(1000, 1., 1000), 1.)

    def test_disabled_single_update_and_zero_peak(self):
        for step in [0, 1, 2, 100]:
            self.assertEqual(kl_lambda_at_step(step, 2., 0), 2.)
            self.assertEqual(kl_lambda_at_step(step, 0., 10), 0.)
        self.assertEqual(kl_lambda_at_step(0, 2., 1), 0.)
        self.assertEqual(kl_lambda_at_step(1, 2., 1), 2.)
        self.assertEqual(kl_lambda_at_step(2, 2., 1), 2.)

    def test_resume_uses_global_update(self):
        uninterrupted = [kl_lambda_at_step(step, 3., 8) for step in range(1, 13)]
        resumed = [kl_lambda_at_step(step, 3., 8) for step in range(6, 13)]
        self.assertEqual(resumed, uninterrupted[5:])

    def test_cli_defaults_and_negative_ramp(self):
        with patch('sys.argv', ['train.py', '--target-kl', '1']):
            self.assertEqual(arguments().kl_steps, 0)
        with patch('sys.argv', ['train.py', '--target-kl', '1', '--kl-steps', '1000']):
            self.assertEqual(arguments().kl_steps, 1000)
        with patch('sys.argv', ['train.py', '--target-kl', '1', '--kl-steps', '-1']), patch('sys.stderr', io.StringIO()):
            with self.assertRaises(SystemExit):
                arguments()
        with self.assertRaises(ValueError):
            kl_lambda_at_step(1, 1., -1)

    def test_schedule_changes_only_training_penalty_and_validation_uses_peak(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = WorldModel(ModelConfig(8, context=2, width=16, heads=2, compressor_layers=1, predictor_layers=1, action_dim=4, dropout=0.))
        idx = torch.tensor([[1, 2]])
        targets = torch.tensor([[2, 3]])
        mean = torch.ones(1, 2, 4, requires_grad=True)
        log_std = torch.zeros_like(mean)
        with patch.object(model, 'posterior', return_value=(mean, log_std)):
            for step, expected_penalty in [(1, 0.), (2, 1.), (3, 2.), (4, 2.)]:
                strength = kl_lambda_at_step(step, 2., 3)
                trained = model(idx, targets, target_kl=1., kl_lambda=strength)
                self.assertEqual(trained['kl_total'].item(), 2.)
                self.assertEqual(trained['kl_penalty'].item(), expected_penalty)
                # Validation uses the configured peak, independently of warmup.
                with torch.no_grad():
                    validated = model(idx, targets, target_kl=1., kl_lambda=2.)
                self.assertEqual(validated['kl_penalty'].item(), 2.)


if __name__ == '__main__':
    unittest.main()
