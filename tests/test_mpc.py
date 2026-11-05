"""Regression tests for compressor-updated, goal-only MPC."""
import unittest
import torch
from llwm import ModelConfig, WorldModel
from llwm.planning import mpc_plan


class MPCTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        self.model = WorldModel(ModelConfig(
            8, context=8, width=16, heads=2, compressor_layers=1,
            predictor_layers=1, action_dim=4, dropout=0.))
        self.prompt = torch.tensor([[0, 1, 2]])
        self.goal = torch.tensor([3, 4])

    def test_receding_horizon_and_logging(self):
        self.model.train()
        result = mpc_plan(self.model, self.prompt, self.goal,
                          horizon=7, execute_steps=3, iterations=1, seed=7)
        self.assertEqual(result['tokens'].shape, (1, 7))
        self.assertEqual([r['horizon'] for r in result['round_history']], [7, 4, 1])
        self.assertEqual([r['executed'] for r in result['round_history']], [3, 3, 1])
        self.assertEqual(result['initial_loss'], result['round_history'][0]['initial_loss'])
        self.assertTrue(all(r['goal_ce'] <= r['initial_loss'] for r in result['round_history']))
        self.assertTrue(self.model.training)
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            mpc_plan(self.model, self.prompt, self.goal, horizon=4, execute_steps=0)
        with self.assertRaises(ValueError):
            mpc_plan(self.model, self.prompt, self.goal, horizon=4, lr=float('nan'))


if __name__ == '__main__':
    unittest.main()
