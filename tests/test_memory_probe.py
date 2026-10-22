import copy
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from llwm import ModelConfig, WorldModel
from llwm.planning import plan
from train import clip_training_gradients, load_training_weights


class MemoryProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = WorldModel(ModelConfig(8, context=4, width=16, heads=2,
                                           compressor_layers=1, predictor_layers=1,
                                           action_dim=4, dropout=0.))
        self.idx = torch.tensor([[1, 2, 3, 4]])
        self.targets = torch.tensor([[2, 3, 4, -1]])

    def test_probe_decodes_detached_current_memory_and_only_updates_itself(self):
        seen = []
        hook = self.model.memory_probe.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
        try:
            values = self.model(self.idx, self.targets, target_kl=1.)
        finally:
            hook.remove()
        self.assertEqual(len(seen), 1)
        self.assertFalse(seen[0].requires_grad)
        memory, _ = self.model.encode(self.idx)
        torch.testing.assert_close(seen[0], memory)
        logits = self.model.memory_probe(memory.detach()).float()
        logits = 15*torch.tanh(logits/15)
        expected = F.cross_entropy(logits.flatten(0, 1), self.targets.flatten(), ignore_index=-1)
        torch.testing.assert_close(values['probe_ce'], expected)
        accuracy = logits[:, :3].argmax(-1).eq(self.targets[:, :3]).float().mean()
        torch.testing.assert_close(values['probe_accuracy'], accuracy)
        values['probe_ce'].backward()
        for name, param in self.model.named_parameters():
            if name.startswith('memory_probe.'):
                self.assertIsNotNone(param.grad, name)
                self.assertGreater(param.grad.abs().sum().item(), 0.)
            else:
                self.assertIsNone(param.grad, name)

    def test_adding_probe_backward_does_not_change_any_main_model_gradient(self):
        values = self.model(self.idx, self.targets, target_kl=0., eqm_weight=0.2, eqm_chunk_size=2)
        values['loss'].backward(retain_graph=True)
        expected = {name: param.grad.clone() for name, param in self.model.named_parameters()
                    if not name.startswith('memory_probe.')}
        self.model.zero_grad(set_to_none=True)
        (values['loss']+values['probe_ce']).backward()
        for name, param in self.model.named_parameters():
            if name in expected:
                torch.testing.assert_close(param.grad, expected[name], msg=name)

    def test_gradient_clipping_is_independent_of_probe_gradient_size(self):
        baseline = copy.deepcopy(self.model)
        for model in [baseline, self.model]:
            for name, param in model.named_parameters():
                param.grad = torch.full_like(param, 1e6 if name.startswith('memory_probe.') else 1.)
        main = [p for name, p in baseline.named_parameters() if not name.startswith('memory_probe.')]
        expected_norm = torch.nn.utils.clip_grad_norm_(main, 1.)
        main_norm, _ = clip_training_gradients(self.model)
        torch.testing.assert_close(main_norm, expected_norm)
        for (name, param), (_, expected) in zip(self.model.named_parameters(), baseline.named_parameters()):
            if not name.startswith('memory_probe.'):
                torch.testing.assert_close(param.grad, expected.grad, msg=name)

    def test_old_checkpoints_preserve_main_weights_and_adam_states(self):
        main = [p for name, p in self.model.named_parameters() if not name.startswith('memory_probe.')]
        legacy_optimizer = torch.optim.AdamW(main, lr=0.01)
        for param in main:
            param.grad = torch.ones_like(param)
        legacy_optimizer.step()
        checkpoint = {'model': {k: v for k, v in self.model.state_dict().items() if not k.startswith('memory_probe.')},
                      'optimizer': legacy_optimizer.state_dict()}
        target = WorldModel(self.model.config)
        probe_before = copy.deepcopy(target.memory_probe.state_dict())
        optimizer = torch.optim.AdamW(target.parameters(), lr=0.01)
        self.assertTrue(load_training_weights(target, optimizer, checkpoint))
        for name, param in target.state_dict().items():
            if name in checkpoint['model']:
                torch.testing.assert_close(param, checkpoint['model'][name])
        for name, value in target.memory_probe.state_dict().items():
            torch.testing.assert_close(value, probe_before[name])
        restored_main = [p for name, p in target.named_parameters() if not name.startswith('memory_probe.')]
        for old, restored in zip(main, restored_main):
            for key, value in legacy_optimizer.state[old].items():
                torch.testing.assert_close(optimizer.state[restored][key], value)
        self.assertTrue(all(p not in optimizer.state for p in target.memory_probe.parameters()))
        # Old serialized groups aren't mutated by compatibility loading.
        self.assertEqual(len(checkpoint['optimizer']['param_groups'][0]['params']), len(main))

    def test_new_checkpoint_restores_probe_and_incomplete_probe_is_rejected(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        values = self.model(self.idx, self.targets, target_kl=1.)
        (values['loss']+values['probe_ce']).backward()
        optimizer.step()
        checkpoint = {'model': self.model.state_dict(), 'optimizer': optimizer.state_dict()}
        target = WorldModel(self.model.config)
        restored = torch.optim.AdamW(target.parameters(), lr=0.01)
        self.assertFalse(load_training_weights(target, restored, checkpoint))
        for key, value in target.state_dict().items():
            torch.testing.assert_close(value, checkpoint['model'][key])
        for old, new in zip(self.model.parameters(), target.parameters()):
            for key, value in optimizer.state[old].items():
                torch.testing.assert_close(restored.state[new][key], value)
        incomplete = dict(checkpoint, model=dict(checkpoint['model']))
        del incomplete['model']['memory_probe.bias']
        with self.assertRaises(RuntimeError):
            load_training_weights(target, restored, incomplete)

    def test_ignored_batch_is_zero_and_planning_does_not_use_probe(self):
        values = self.model(self.idx, torch.full_like(self.targets, -1), target_kl=1.)
        self.assertEqual(values['probe_ce'].item(), 0.)
        self.assertEqual(values['probe_accuracy'].item(), 0.)
        values['probe_ce'].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.memory_probe.parameters()))
        with patch.object(self.model.memory_probe, 'forward', side_effect=AssertionError('planning must not use probe')):
            result = plan(self.model, self.idx, torch.tensor([2]), horizon=3, iterations=2)
        self.assertEqual(result['tokens'].shape, (1, 3))
