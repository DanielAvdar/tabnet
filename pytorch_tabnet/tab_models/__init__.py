"""TabNet models sub-package.

This package contains the TabNet model implementations including:
- TabNetClassifier
- TabNetRegressor
- TabNetMultiTaskClassifier
- TabNetPretrainer

These models implement the TabNet architecture as described in the paper
"TabNet: Attentive Interpretable Tabular Learning".
"""

from .abstract_models import TabSupervisedModel
from .multitask import TabNetMultiTaskClassifier
from .pretraining import TabNetPretrainer
from .tab_class import TabNetClassifier
from .tab_reg import MultiTabNetRegressor, TabNetRegressor

__all__ = [
    "TabSupervisedModel",
    "TabNetClassifier",
    "TabNetRegressor",
    "MultiTabNetRegressor",
    "TabNetPretrainer",
    "TabNetMultiTaskClassifier",
]
