import numpy as np
import pytest

from pytorch_tabnet.multiclass_utils.type_detection import check_classification_targets


def test_check_classification_targets_binary():
    """Test check_classification_targets with binary targets."""
    y = np.array([0, 1, 0, 1])
    # Should not raise any exception
    check_classification_targets(y)


def test_check_classification_targets_multiclass():
    """Test check_classification_targets with multiclass targets."""
    y = np.array([0, 1, 2, 3])
    # Should not raise any exception
    check_classification_targets(y)


def test_check_classification_targets_multilabel():
    """Test check_classification_targets with multilabel targets."""
    y = np.array([[0, 1], [1, 0], [1, 1]])
    # Should not raise any exception
    check_classification_targets(y)


def test_check_classification_targets_multiclass_multioutput():
    """Test check_classification_targets with multiclass-multioutput targets."""
    y = np.array([[0, 1], [1, 0], [2, 3]])
    # Should not raise any exception
    check_classification_targets(y)


def test_check_classification_targets_continuous():
    """Test check_classification_targets with continuous targets raises error."""
    y = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="Unknown label type"):
        check_classification_targets(y)
