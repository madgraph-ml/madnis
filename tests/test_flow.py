"""Tests for Normal distributions"""

import unittest

import torch
import torchtestcase

from madnis.nn.flow import Flow


# TODO: Add more. Just to have a starter
class FlowTest(torchtestcase.TorchTestCase):
    def test_log_prob(self):
        batch_size = 10
        input_dim = 5
        flow = Flow(input_dim)
        inputs = torch.randn(batch_size, input_dim)
        log_prob = flow.log_prob(inputs)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(log_prob.shape, torch.Size([batch_size]))
        self.assertFalse(torch.isnan(log_prob).any())
        self.assertFalse(torch.isinf(log_prob).any())

    def test_prob(self):
        batch_size = 10
        input_dim = 5
        flow = Flow(input_dim)
        inputs = torch.randn(batch_size, input_dim)
        prob = flow.prob(inputs)
        log_prob = flow.log_prob(inputs)
        self.assertIsInstance(prob, torch.Tensor)
        self.assertEqual(torch.exp(log_prob), prob)

    def test_transform(self):
        self.eps = 1e-8
        batch_size = 10
        input_dim = 5
        flow = Flow(input_dim)
        inputs = torch.rand(batch_size, input_dim)
        forward, fw_jac = flow.transform(inputs)
        inverse, inv_jac = flow.transform(forward, inverse=True)
        forward2, _ = flow.transform(inverse)
        self.assertIsInstance(forward, torch.Tensor)
        self.assertEqual(inverse, inputs)
        self.assertEqual(forward, forward2)
        self.assertEqual(fw_jac, -inv_jac)


if __name__ == "__main__":
    unittest.main()
