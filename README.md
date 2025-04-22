# TabNet: Attentive Interpretable Tabular Learning

![PyPI version](https://img.shields.io/pypi/v/eh-pytorch-tabnet.svg)
![Python versions](https://img.shields.io/pypi/pyversions/eh-pytorch-tabnet.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![OS](https://img.shields.io/badge/ubuntu-blue?logo=ubuntu)
![OS](https://img.shields.io/badge/win-blue?logo=windows)
![OS](https://img.shields.io/badge/mac-blue?logo=apple)
[![codecov](https://codecov.io/gh/DanielAvdar/tabnet/graph/badge.svg?token=N0V9KANTG2)](https://codecov.io/gh/DanielAvdar/tabnet)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Last Commit](https://img.shields.io/github/last-commit/DanielAvdar/tabnet/main)


TabNet is a deep learning architecture designed specifically for tabular data,
combining interpretability and high predictive performance.
This package provides a modern, maintained implementation of TabNet in PyTorch,
supporting classification, regression, multitask learning, and unsupervised pretraining.


## Installation

Install TabNet using pip:

```bash
pip install eh-pytorch-tabnet
```

## What is TabNet?
TabNet is an interpretable neural network architecture for tabular data, introduced by Arik & Pfister (2019). It uses sequential attention to select which features to reason from at each decision step, enabling both high performance and interpretability. TabNet learns sparse feature masks, allowing users to understand which features are most important for each prediction. The method is particularly effective for structured/tabular datasets where traditional deep learning models often underperform compared to tree-based methods.

Key aspects of TabNet:
- **Attentive Feature Selection**: At each step, TabNet learns which features to focus on, improving both accuracy and interpretability.
- **Interpretable Masks**: The model produces feature masks that highlight the importance of each feature for individual predictions.
- **End-to-End Learning**: Supports classification, regression, multitask, and unsupervised pretraining tasks.

# What problems does pytorch-tabnet handle?

- TabNetClassifier : binary classification and multi-class classification problems
- TabNetRegressor : simple and multi-task regression problems
- TabNetMultiTaskClassifier:  multi-task multi-classification problems


## Usage

### [Documentation](https://eh-pytorch-tabnet.readthedocs.io/en/latest/)


### Examples

```python
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

clf = TabNetClassifier()
clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
preds = clf.predict(X_test)
```

For multitask classification:
```python
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
clf = TabNetMultiTaskClassifier()
clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
preds = clf.predict(X_test)
```

For pretraining:
```python
from pytorch_tabnet.pretraining import TabNetPretrainer
pretrainer = TabNetPretrainer()
pretrainer.fit(X_train)
```

See the [nbs/](nbs/) folder for more complete examples and notebooks.

## Further Reading
- [TabNet: Attentive Interpretable Tabular Learning (Arik & Pfister, 2019)](https://arxiv.org/pdf/1908.07442.pdf)
- Original repo: https://github.com/dreamquark-ai/tabnet

## License & Credits
- Original implementation and research by DreamQuark team
- Maintained and improved by Daniel Avdar and contributors
- See LICENSE for details
