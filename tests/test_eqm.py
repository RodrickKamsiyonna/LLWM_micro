import unittest
from unittest.mock import patch
import torch
from llwm import ModelConfig, WorldModel
from llwm.eqm import chunk_loss, equilibrium_matching


class EQMTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = WorldModel(ModelConfig(8, context=8, width=16, heads=2, compressor_layers=1, predictor_layers=1, action_dim=4, dropout=0.))
        self.tokens = torch.randint(8, (2, 7))

    def test_exact_value_and_all_parameter_gradients_match_reference(self):
        memory = self.model.compress(self.tokens)[:, :-1]
        action = torch.randn(2, 6, 4, requires_grad=True)
        targets = self.tokens[:, 1:].clone()
        targets[0, 0] = -1
        state = torch.get_rng_state()
        actual = equilibrium_matching(self.model, memory, action, targets, strength=1.3, chunk_size=5)
        params = tuple(self.model.parameters())
        actual_grads = torch.autograd.grad(actual, params + (action,), allow_unused=True)
        torch.set_rng_state(state)
        gamma = torch.rand(2, 6, 1)
        noise = torch.randn_like(action)
        reference_memory = self.model.compress(self.tokens)[:, :-1]
        expected = chunk_loss(self.model, reference_memory.reshape(-1, 16), action.reshape(-1, 4),
                              targets.reshape(-1), gamma.reshape(-1, 1), noise.reshape(-1, 4), 1.3, True)
        expected_grads = torch.autograd.grad(expected, params + (action,), allow_unused=True)
        torch.testing.assert_close(actual, expected)
        for (name, _), a, b in zip(list(self.model.named_parameters()) + [('action', action)], actual_grads, expected_grads):
            self.assertEqual(a is None, b is None, name)
            if a is not None:
                torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-4, msg=name)
        self.assertIsNone(actual_grads[-1])
        self.assertGreater(actual_grads[0].abs().sum().item(), 0.)

    def test_no_grad_evaluation_and_loss_scaling(self):
        with torch.no_grad():
            evaluated = self.model(self.tokens, eqm_chunk_size=3)
        self.assertTrue(torch.isfinite(evaluated['eqm']))
        self.assertFalse(evaluated['eqm'].requires_grad)
        state = torch.get_rng_state()
        a = self.model(self.tokens, eqm_chunk_size=3)
        a['eqm'].backward()
        gradients = [p.grad.clone() if p.grad is not None else None for p in self.model.parameters()]
        self.model.zero_grad(set_to_none=True)
        torch.set_rng_state(state)
        b = self.model(self.tokens, eqm_chunk_size=3)
        (b['eqm'] * 128).backward()
        for p, expected in zip(self.model.parameters(), gradients):
            if expected is not None:
                torch.testing.assert_close(p.grad, expected * 128)

    def test_chunk_size_preserves_objective_and_eqm_can_be_disabled(self):
        memory = self.model.compress(self.tokens)[:, :-1]
        action = torch.randn(2, 6, 4)
        state = torch.get_rng_state()
        a = equilibrium_matching(self.model, memory, action, self.tokens[:, 1:], chunk_size=1)
        torch.set_rng_state(state)
        b = equilibrium_matching(self.model, memory, action, self.tokens[:, 1:], chunk_size=100)
        torch.testing.assert_close(a, b)
        disabled = self.model(self.tokens, eqm_weight=0.)
        self.assertEqual(disabled['eqm'].item(), 0.)

    def test_saved_tensor_hooks_preserve_eqm_gradients(self):
        state = torch.get_rng_state()
        actual = self.model(self.tokens, eqm_chunk_size=3)
        actual['eqm'].backward()
        expected = [p.grad.clone() if p.grad is not None else None for p in self.model.parameters()]
        self.model.zero_grad(set_to_none=True)
        torch.set_rng_state(state)
        with torch.autograd.graph.saved_tensors_hooks(lambda t: t.detach(), lambda t: t):
            hooked = self.model(self.tokens, eqm_chunk_size=3)
            hooked['eqm'].backward()
        for (name, p), gradient in zip(self.model.named_parameters(), expected):
            self.assertEqual(p.grad is None, gradient is None, name)
            if gradient is not None:
                torch.testing.assert_close(p.grad, gradient, msg=name)

    def test_disabled_eqm_is_never_called_in_training_or_validation(self):
        with patch('llwm.model.equilibrium_matching', side_effect=AssertionError('EQM must not run')) as eqm:
            trained = self.model(self.tokens, eqm_weight=0.)
            trained['loss'].backward()
            self.model.eval()
            with torch.no_grad():
                evaluated = self.model(self.tokens, eqm_weight=0.)
            eqm.assert_not_called()
        for values in [trained, evaluated]:
            self.assertEqual(values['eqm'].item(), 0.)
            self.assertFalse(values['eqm'].requires_grad)
            torch.testing.assert_close(values['loss'], values['ce'] + values['mse'] + 0.001*values['kl'])

    def test_disable_eqm_cli_overrides_weight(self):
        from train import arguments
        with patch('sys.argv', ['train.py', '--disable-eqm', '--eqm-weight', '3']):
            args = arguments()
        self.assertTrue(args.disable_eqm)
        self.assertEqual(args.eqm_weight, 0.)


if __name__ == '__main__':
    unittest.main()
