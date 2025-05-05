"""pytorch_tabnet package initialization."""

from importlib.metadata import version

from .multitask import TabNetMultiTaskClassifier
from .pretraining import TabNetPretrainer
from .tab_model import MultiTabNetRegressor, TabNetClassifier, TabNetRegressor

__version__ = version("pytorch-tabnet2")
