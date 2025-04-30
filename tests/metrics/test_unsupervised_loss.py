"""Tests for unsupervised_loss.py module."""

import unittest

import torch

from pytorch_tabnet.metrics.unsupervised_loss import UnsupervisedLoss


class TestUnsupervisedLoss(unittest.TestCase):
    """Tests for the UnsupervisedLoss function."""

    def test_perfect_reconstruction(self):
        """Test with perfect reconstruction."""
        # Setting up test data
        embedded_x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        y_pred = embedded_x.clone()  # Perfect reconstruction
        obf_vars = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        loss = UnsupervisedLoss(y_pred, embedded_x, obf_vars)
        self.assertEqual(loss.item(), 0.0)  # Perfect reconstruction means zero loss

    def test_imperfect_reconstruction(self):
        """Test with imperfect reconstruction."""
        # Setting up test data
        embedded_x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        # Add 1.0 to each value for an imperfect reconstruction
        y_pred = embedded_x + 1.0
        obf_vars = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        loss = UnsupervisedLoss(y_pred, embedded_x, obf_vars)
        self.assertGreater(loss.item(), 0.0)  # Imperfect reconstruction means non-zero loss

    def test_partial_obfuscation(self):
        """Test with partial obfuscation mask."""
        # Setting up test data
        embedded_x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        # Imperfect reconstruction
        y_pred = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
        # Only the first column is obfuscated and should be reconstructed
        obf_vars = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        loss = UnsupervisedLoss(y_pred, embedded_x, obf_vars)

        # Calculate expected loss manually - only first column matters
        errors = (y_pred - embedded_x) ** 2
        masked_errors = errors * obf_vars
        batch_std = torch.std(embedded_x, dim=0)[0] ** 2
        if batch_std == 0:
            batch_std = torch.mean(embedded_x, dim=0)[0]
        expected_loss = torch.mean(masked_errors[:, 0] / batch_std)

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)

    def test_with_weights(self):
        """Test with sample weights."""
        # Setting up test data
        embedded_x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        # Imperfect reconstruction
        y_pred = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
        obf_vars = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        # Define weights for each sample
        weights = torch.tensor([2.0, 1.0, 0.5])

        loss = UnsupervisedLoss(y_pred, embedded_x, obf_vars, weights=weights)

        # Try without weights for comparison
        unweighted_loss = UnsupervisedLoss(y_pred, embedded_x, obf_vars)

        # The weighted loss should be different than the unweighted loss
        self.assertNotEqual(loss.item(), unweighted_loss.item())

    def test_zero_std_handling(self):
        """Test the handling of zero standard deviation."""
        # Setting up test data with all same values in one column (zero std)
        embedded_x = torch.tensor([[1.0, 2.0, 5.0], [1.0, 3.0, 5.0], [1.0, 4.0, 5.0]])
        # Imperfect reconstruction
        y_pred = torch.tensor([[2.0, 3.0, 6.0], [2.0, 4.0, 6.0], [2.0, 5.0, 6.0]])
        obf_vars = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        # This should not raise an error despite zero std in first column
        loss = UnsupervisedLoss(y_pred, embedded_x, obf_vars)

        # Verify it's a non-zero, finite value
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss))
