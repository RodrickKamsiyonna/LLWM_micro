import copy
import io
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from llwm import ModelConfig, WorldModel
from llwm.eqm import equilibrium_matching_loss
from train import arguments


class EQMTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = WorldModel(ModelConfig(8, context=4, width=16, heads=2,
                                           compressor_layers=1, predictor_layers=1,
                                           action_dim=4, dropout=0.))
        self.idx = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
        self.targets = torch.tensor([[2, 3, 4, 5], [3, 4, 5, -1]])

    def test_chunked_values_and_all_gradients_match_full_second_order_reference(self):
        reference = copy.deepcopy(self.model)
        action = torch.randn(2, 4, 4, requires_grad=True)
        rng = torch.get_rng_state()
        h, _ = reference.encode(self.idx)
        valid = self.targets.flatten().ne(-1)
        a = action.detach().reshape(-1, 4)[valid]
        torch.set_rng_state(rng)
        gamma = torch.rand(a.size(0), 1)
        eps = torch.randn_like(a)
        noisy = (gamma*a+(1-gamma)*eps).detach().requires_grad_(True)
        logits = reference.predict_logits(h.reshape(-1, 16)[valid], noisy)
        energy = F.cross_entropy(logits, self.targets.flatten()[valid], reduction='sum')
        gradient, = torch.autograd.grad(energy, noisy, create_graph=True)
        expected = (gradient-(eps-a)*0.7*(1-gamma)).square().mean()
        expected.backward()
        for size in [1, 3, 0]:
            with self.subTest(chunk_size=size):
                self.model.zero_grad(set_to_none=True)
                memory, _ = self.model.encode(self.idx)
                torch.set_rng_state(rng)
                actual = equilibrium_matching_loss(self.model, memory, action, self.targets,
                                                   eqm_lambda=0.7, chunk_size=size)
                torch.testing.assert_close(actual, expected)
                actual.backward()
                for (name, param), (_, ref_param) in zip(self.model.named_parameters(), reference.named_parameters()):
                    if ref_param.grad is None:
                        self.assertIsNone(param.grad, name)
                    else:
                        torch.testing.assert_close(param.grad, ref_param.grad, atol=2e-6, rtol=2e-4, msg=name)
                self.assertGreater(self.model.embedding.weight.grad.abs().sum().item(), 0.)
                self.assertIsNone(action.grad)

    def test_forward_eqm_updates_memory_predictor_head_but_not_action_encoder(self):
        values = self.model(self.idx, self.targets, target_kl=1., eqm_weight=0.3, eqm_chunk_size=3)
        torch.testing.assert_close(values['eqm_penalty'], 0.3*values['eqm'])
        torch.testing.assert_close(values['loss'], values['ce']+values['memory_loss']+values['kl_penalty']+values['eqm_penalty'])
        values['eqm'].backward()
        for module in [self.model.compressor, self.model.predictor, self.model.head]:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.action_encoder.parameters()))

    def test_disabled_skip_and_ignored_positions(self):
        with patch('llwm.model.equilibrium_matching_loss', side_effect=AssertionError('disabled')):
            values = self.model(self.idx, self.targets, target_kl=1.)
        self.assertEqual(values['eqm'].item(), 0.)
        self.assertEqual(values['eqm_penalty'].item(), 0.)
        h, _ = self.model.encode(self.idx)
        action = torch.randn(2, 4, 4, requires_grad=True)
        value = equilibrium_matching_loss(self.model, h, action, torch.full_like(self.targets, -1), chunk_size=2)
        self.assertEqual(value.item(), 0.)
        value.backward()
        self.assertIsNone(action.grad)

    def test_validation_and_autocast(self):
        with torch.no_grad():
            value = self.model(self.idx, self.targets, target_kl=1., eqm_weight=1., eqm_chunk_size=2)
        self.assertFalse(value['eqm'].requires_grad)
        self.assertTrue(torch.isfinite(value['eqm']))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            value = self.model(self.idx, self.targets, target_kl=1., eqm_weight=1., eqm_chunk_size=2)
        (value['loss']+value['probe_ce']).backward()
        self.assertEqual(value['eqm'].dtype, torch.float32)
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)

    def test_cli_settings_and_invalid_values(self):
        with patch('sys.argv', ['train.py', '--target-kl', '1']):
            args = arguments()
        self.assertEqual((args.eqm_weight, args.eqm_lambda, args.eqm_chunk_size), (0., 1., 128))
        for option, value in [('--eqm-weight', '-1'), ('--eqm-lambda', 'nan'), ('--eqm-chunk-size', '-1')]:
            with patch('sys.argv', ['train.py', '--target-kl', '1', option, value]), patch('sys.stderr', io.StringIO()):
                with self.assertRaises(SystemExit):
                    arguments()

    def test_outer_scaling_and_saved_tensor_hooks_preserve_gradients(self):
        state = torch.get_rng_state()
        result = self.model(self.idx, self.targets, target_kl=1., eqm_weight=1., eqm_chunk_size=3)
        result['eqm'].backward()
        expected = [p.grad.clone() if p.grad is not None else None for p in self.model.parameters()]
        self.model.zero_grad(set_to_none=True)
        torch.set_rng_state(state)
        with torch.autograd.graph.saved_tensors_hooks(lambda t: t.detach(), lambda t: t):
            result = self.model(self.idx, self.targets, target_kl=1., eqm_weight=1., eqm_chunk_size=3)
            (128*result['eqm']).backward()
        for (name, param), gradient in zip(self.model.named_parameters(), expected):
            self.assertEqual(param.grad is None, gradient is None, name)
            if gradient is not None:
                torch.testing.assert_close(param.grad, 128*gradient, msg=name)

    def test_recomputation_reduces_peak_saved_storage(self):
        from benchmark_eqm import SavedStorage
        from llwm.eqm import chunk_loss
        model = WorldModel(ModelConfig(16, width=64, heads=4, compressor_layers=1,
                                       predictor_layers=3, action_dim=16, dropout=0.))
        memory = torch.randn(1, 256, 64, requires_grad=True)
        action = torch.randn(1, 256, 16)
        targets = torch.randint(16, (1, 256))
        peaks, values = [], []
        for recomputed in [False, True]:
            model.zero_grad(set_to_none=True)
            memory.grad = None
            tracker = SavedStorage()
            torch.manual_seed(23)
            with torch.autograd.graph.saved_tensors_hooks(tracker.pack, lambda held: held.tensor):
                if recomputed:
                    loss = equilibrium_matching_loss(model, memory, action, targets, chunk_size=32)
                else:
                    gamma = torch.rand(256, 1)
                    noise = torch.randn(256, 16)
                    loss = chunk_loss(model, memory.reshape(-1, 64), action.reshape(-1, 16),
                                      targets.flatten(), gamma, noise, 1., True)
                values.append(loss.detach())
                loss.backward()
            peaks.append(tracker.peak)
        torch.testing.assert_close(values[0], values[1])
        self.assertLess(peaks[1], peaks[0]/2)
