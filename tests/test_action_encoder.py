import io
import unittest
from unittest.mock import patch

import torch
from torch import nn

from llwm import ModelConfig, WorldModel
from llwm.model import ActionEncoder
from train import arguments, load_training_weights


class ActionEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_default_progressive_halving_and_separate_gaussian_heads(self):
        encoder = ActionEncoder(256, 32, num_layers=4)
        layers = [layer for layer in encoder.trunk if isinstance(layer, nn.Linear)]
        self.assertEqual([(layer.in_features, layer.out_features) for layer in layers],
                         [(512, 256), (256, 128), (128, 64), (64, 32)])
        self.assertEqual(sum(isinstance(layer, nn.GELU) for layer in encoder.trunk), 4)
        output = encoder(torch.randn(2, 3, 512))
        self.assertEqual(output.shape, (2, 3, 64))
        self.assertEqual(encoder.mean.in_features, 32)
        self.assertEqual(encoder.mean.out_features, 32)
        self.assertEqual(encoder.log_std.out_features, 32)
        output.square().mean().backward()
        for name, parameter in encoder.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_non_power_of_two_width_stops_at_action_dimension(self):
        encoder = ActionEncoder(24, 5, num_layers=4)
        self.assertEqual([layer.out_features for layer in encoder.trunk if isinstance(layer, nn.Linear)],
                         [24, 12, 6, 5])
        self.assertEqual(encoder(torch.randn(2, 48)).shape, (2, 10))

    def test_equal_width_and_invalid_action_dimensions(self):
        self.assertEqual(ActionEncoder(8, 16)(torch.randn(2, 16)).shape, (2, 32))
        for action_dim in [0, -1, 17]:
            with self.assertRaises(ValueError):
                ActionEncoder(8, action_dim)

    def test_fixed_training_dimension_and_configurable_depth(self):
        self.assertEqual(ModelConfig(8).action_dim, 128)
        for depth, expected in [(1, [128]), (2, [256, 128]), (4, [256, 128, 128, 128])]:
            with self.subTest(depth=depth):
                encoder = ActionEncoder(256, num_layers=depth)
                self.assertEqual([layer.out_features for layer in encoder.trunk if isinstance(layer, nn.Linear)], expected)
                self.assertEqual(encoder(torch.randn(1, 512)).shape, (1, 256))
        with patch('sys.argv', ['train.py', '--target-kl', '1']):
            args = arguments()
        self.assertEqual((args.action_dim, args.action_layers), (128, 2))
        with patch('sys.argv', ['train.py', '--target-kl', '1', '--action-layers', '4']):
            self.assertEqual(arguments().action_layers, 4)

    def test_invalid_depth_and_fixed_dimension_cli(self):
        for depth in [0, -1]:
            with self.assertRaises(ValueError):
                ActionEncoder(256, num_layers=depth)
        for option, value in [('--action-layers', '0'), ('--action-layers', '-1'), ('--action-dim', '32'), ('--width', '32')]:
            with patch('sys.argv', ['train.py', '--target-kl', '1', option, value]), patch('sys.stderr', io.StringIO()):
                with self.assertRaises(SystemExit):
                    arguments()

    def test_model_routes_128_dimensional_actions_through_training_and_planning(self):
        from llwm.planning import plan
        model = WorldModel(ModelConfig(8, width=64, heads=4, context=4,
                                       compressor_layers=1, predictor_layers=1,
                                       action_layers=3, dropout=0.))
        idx, targets = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])
        memory, _ = model.encode(idx)
        mean, log_std = model.posterior(memory, memory)
        self.assertEqual(mean.shape, (1, 3, 128))
        self.assertEqual(log_std.shape, mean.shape)
        values = model(idx, targets, target_kl=1., eqm_weight=0.1, eqm_chunk_size=2)
        (values['loss']+values['probe_ce']).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        result = plan(model, idx, torch.tensor([2]), horizon=3, iterations=2)
        self.assertEqual(result['tokens'].shape, (1, 3))

    def test_old_two_layer_checkpoint_is_rejected_with_clear_message(self):
        model = WorldModel(ModelConfig(8, width=16, heads=2, compressor_layers=1,
                                       predictor_layers=1, action_dim=4))
        old = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 8))
        weights = {key: value for key, value in model.state_dict().items() if not key.startswith('action_encoder.')}
        weights.update({f'action_encoder.{key}': value for key, value in old.state_dict().items()})
        optimizer = torch.optim.AdamW(model.parameters())
        with self.assertRaisesRegex(ValueError, 'halving-width architecture'):
            load_training_weights(model, optimizer, {'model': weights, 'optimizer': optimizer.state_dict()})
