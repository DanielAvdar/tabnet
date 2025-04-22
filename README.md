# TabNet : Attentive Interpretable Tabular Learning

![PyPI version](https://img.shields.io/pypi/v/eh-pytorch-tabnet.svg)
![Python versions](https://img.shields.io/pypi/pyversions/eh-pytorch-tabnet.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Ubuntu](https://img.shields.io/badge/ubuntu-blue?logo=ubuntu)
![Windows](https://img.shields.io/badge/ubuntu-blue?logo=windows)
![MacOS](https://img.shields.io/badge/ubuntu-blue?logo=apple)
![Coverage](https://codecov.io/gh/DanielAvdar/eh-pytorch-tabnet/graph/badge.svg?token=N0V9KANTG2)
![Last Commit](https://img.shields.io/github/last-commit/DanielAvdar/eh-pytorch-tabnet/main)

This is a maintained fork of [`dreamquark-ai/tabnet`](https://github.com/dreamquark-ai/tabnet) with improvements and changes for modern PyTorch, metrics, and GPU support.

## Key Features
- Supports classification, regression, multitask, and unsupervised pretraining
- GPU acceleration and efficient data handling
- Interpretable feature masks and explanations
- Flexible API for research and production

## Changes from the Original Implementation
- **PyTorch Metrics**: Uses PyTorch-based metrics for better GPU compatibility and performance
- **Enhanced GPU Support**: Improved prediction and evaluation logic for efficient CUDA (GPU) execution
- **API Improvements**: More flexible and user-friendly API for model training, evaluation, and prediction
- **Bug Fixes and Maintenance**: Ongoing fixes and updates to ensure compatibility with recent PyTorch and Python versions
- **Documentation and Examples**: Expanded documentation and new example notebooks for easier onboarding and advanced usage
- **License and Policy**: All code is considered changed for license clarity; see LICENSE for details

These changes aim to make TabNet more robust, performant, and accessible for both research and production use cases.

## Original Paper & Reference
- [TabNet: Attentive Interpretable Tabular Learning (Arik & Pfister, 2019)](https://arxiv.org/pdf/1908.07442.pdf)
- Original repo: https://github.com/dreamquark-ai/tabnet

## Example Usage
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

## License & Credits
- Original implementation and research by DreamQuark team
- Maintained and improved by Daniel Avdar and contributors
- See LICENSE for details
