import io
import unittest
from unittest.mock import patch

import torch

from llwm import ModelConfig, WorldModel
from train import arguments


class MemoryLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = WorldModel(ModelConfig(8, context=4, width=16, heads=2,
                                           compressor_layers=1, predictor_layers=1,
                                           action_dim=4, dropout=0.))
        self.idx = torch.tensor([[1, 2, 3, 4]])
        self.targets = torch.tensor([[2, 3, 4, 5]])

    def test_cli_default_and_both_choices(self):
        with patch('sys.argv', ['train.py', '--target-kl', '1']):
            self.assertEqual(arguments().memory_loss, 'smooth_l1')
        for choice in ['mse', 'smooth_l1']:
            with patch('sys.argv', ['train.py', '--target-kl', '1', '--memory-loss', choice]):
                self.assertEqual(arguments().memory_loss, choice)

    def test_invalid_memory_loss_rejected(self):
        with patch('sys.argv', ['train.py', '--target-kl', '1', '--memory-loss', 'invalid']), patch('sys.stderr', io.StringIO()):
            with self.assertRaises(SystemExit):
                arguments()
        with self.assertRaisesRegex(ValueError, 'memory_loss must be'):
            self.model(self.idx, self.targets, target_kl=1., memory_loss='invalid')

    def test_both_losses_have_correct_masked_values_and_gradients(self):
        targets = self.targets.clone()
        targets[:, -1] = -1
        with torch.no_grad():
            states, _ = self.model.encode(torch.cat((self.idx, self.targets[:, -1:]), 1))
            teacher = states[:, 1:]
        mask = targets.ne(-1)[..., None]
        denominator = mask.sum()*self.model.config.width
        original = self.model.predict_features
        for choice in ['mse', 'smooth_l1']:
            with self.subTest(choice=choice):
                predictions = []
                def predict(memory, action):
                    result = original(memory, action)
                    predictions.append(result)
                    return result
                with patch.object(self.model, 'predict_features', side_effect=predict):
                    values = self.model(self.idx, targets, target_kl=1., memory_loss=choice, mse_weight=0.3)
                difference = predictions[0]-teacher
                if choice == 'mse':
                    element_loss, derivative = difference.square(), 2*difference
                else:
                    magnitude = difference.abs()
                    element_loss = torch.where(magnitude < 1, 0.5*difference.square(), magnitude-0.5)
                    derivative = difference.clamp(-1, 1)
                expected = (element_loss*mask).sum()/denominator
                torch.testing.assert_close(values['memory_loss'], expected)
                gradient, = torch.autograd.grad(values['memory_loss'], predictions[0])
                torch.testing.assert_close(gradient, derivative*mask/denominator)
                torch.testing.assert_close(values['loss'], values['ce']+values['kl_penalty']+0.3*expected)
                ignored = self.model(self.idx, torch.full_like(targets, -1), target_kl=1., memory_loss=choice)
                self.assertEqual(ignored['memory_loss'].item(), 0.)
