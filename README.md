# TabNet: Attentive Interpretable Tabular Learning

![PyPI version](https://img.shields.io/pypi/v/pytorch-tabnet2.svg)
![Python versions](https://img.shields.io/pypi/pyversions/pytorch-tabnet2.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![OS](https://img.shields.io/badge/ubuntu-blue?logo=ubuntu)
![OS](https://img.shields.io/badge/win-blue?logo=windows)
![OS](https://img.shields.io/badge/mac-blue?logo=apple)
[![codecov](https://codecov.io/gh/DanielAvdar/tabnet/branch/main/graph/badge.svg)](https://codecov.io/gh/DanielAvdar/tabnet/tree/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Last Commit](https://img.shields.io/github/last-commit/DanielAvdar/tabnet/main)


TabNet is a deep learning architecture designed specifically for tabular data,
combining interpretability and high predictive performance.
This package provides a modern, maintained implementation of TabNet in PyTorch,
supporting classification, regression, multitask learning, and unsupervised pretraining.


## Installation

Install TabNet using pip:

```bash
pip install pytorch-tabnet2
```

## What is TabNet?
TabNet is an interpretable neural network architecture for tabular data, introduced by Arik & Pfister (2019). It uses sequential attention to select which features to reason from at each decision step, enabling both high performance and interpretability. TabNet learns sparse feature masks, allowing users to understand which features are most important for each prediction. The method is particularly effective for structured/tabular datasets where traditional deep learning models often underperform compared to tree-based methods.

Key aspects of TabNet:
- **Attentive Feature Selection**: At each step, TabNet learns which features to focus on, improving both accuracy and interpretability.
- **Interpretable Masks**: The model produces feature masks that highlight the importance of each feature for individual predictions.
- **End-to-End Learning**: Supports classification, regression, multitask, and unsupervised pretraining tasks.

# What problems does pytorch-tabnet handle?

- TabNetClassifier : binary classification and multi-class classification problems.
- TabNetRegressor : simple and multi-task regression problems.
- TabNetMultiTaskClassifier:  multi-task multi-classification problems.
- MultiTabNetRegressor: multi-task regression problems, which is basically TabNetRegressor with multiple targets.


## Usage

### [Documentation](https://tabnet.readthedocs.io/en/latest/)


### Basic Examples

**Classification**
```python
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier

# Generate dummy data
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, 100)
X_valid = np.random.rand(20, 10)
y_valid = np.random.randint(0, 2, 20)
X_test = np.random.rand(10, 10)

clf = TabNetClassifier()
clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
preds = clf.predict(X_test)
print('Predictions:', preds)
```

**Regression**
```python
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor

# Generate dummy data
X_train = np.random.rand(100, 10)
y_train = np.random.rand(100).reshape(-1, 1)
X_valid = np.random.rand(20, 10)
y_valid = np.random.rand(20).reshape(-1, 1)
X_test = np.random.rand(10, 10)

reg = TabNetRegressor()
reg.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
preds = reg.predict(X_test)
print('Predictions:', preds)
```

**Multi-task Classification**
```python
import numpy as np
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier

# Generate dummy data
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, (100, 3))  # 3 tasks
X_valid = np.random.rand(20, 10)
y_valid = np.random.randint(0, 2, (20, 3))
X_test = np.random.rand(10, 10)

clf = TabNetMultiTaskClassifier()
clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
preds = clf.predict(X_test)
print('Predictions:', preds)
```

See the [nbs/](nbs/) folder for more complete examples and notebooks.

## Further Reading
- [TabNet: Attentive Interpretable Tabular Learning (Arik & Pfister, 2019)](https://arxiv.org/pdf/1908.07442.pdf)
- Original repo: https://github.com/dreamquark-ai/tabnet

## License & Credits
- Original implementation and research by [DreamQuark team](https://github.com/dreamquark-ai/tabnet)
- Maintained and improved by Daniel Avdar and contributors
- See LICENSE for details
