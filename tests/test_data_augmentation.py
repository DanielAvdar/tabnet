import torch
import pytest

from pytorch_tabnet.augmentations import RegressionSMOTE, ClassificationSMOTE
from pytorch_tabnet.utils import define_device
import numpy as np

@pytest.fixture
def fix_seed():
    torch.manual_seed(0)
    np.random.seed(0)
    return


@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 2])
@pytest.mark.parametrize("beta", [0.1, 0.5, 2])
def test_regression_smote(fix_seed, p, alpha, beta):

    device = define_device("cpu")  # or your preferred device
    batch_size = 100
    n_features = 10
    X = torch.randn(batch_size, n_features, device=device)
    y = torch.randn(batch_size, 1, device=device)

    smote = RegressionSMOTE(device_name="cpu", p=p, alpha=alpha, beta=beta)
    X_augmented, y_augmented = smote(X.clone(), y.clone())


    if p == 0:
        assert torch.all(X_augmented == X)
        assert torch.all(y_augmented == y)

@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 2])
@pytest.mark.parametrize("beta", [0.1, 0.5, 2])
def test_classification_smote(fix_seed, p, alpha, beta):

    device = define_device("cpu")  # or your preferred device
    batch_size = 100
    n_features = 10
    X = torch.randn(batch_size, n_features, device=device)
    y = torch.randint(0, 2, (batch_size, 1), device=device)  # Example binary labels

    smote = (
        ClassificationSMOTE(device_name="cpu", p=p, alpha=alpha, beta=beta))
    X_augmented, y_augmented = smote(X.clone(), y.clone())

    if p == 0:
        assert torch.all(X_augmented == X)
        assert torch.all(y_augmented == y)

def test_invalid_p_value():
    with pytest.raises(ValueError):
        RegressionSMOTE(p=-0.1)

    with pytest.raises(ValueError):
        RegressionSMOTE(p=1.5)