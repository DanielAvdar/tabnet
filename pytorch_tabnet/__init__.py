"""pytorch_tabnet package initialization."""

from importlib.metadata import version

from .multitask import TabNetMultiTaskClassifier as TabNetMultiTaskClassifier
from .pretraining import TabNetPretrainer as TabNetPretrainer
from .tab_model import MultiTabNetRegressor as MultiTabNetRegressor
from .tab_model import TabNetClassifier as TabNetClassifier
from .tab_model import TabNetRegressor as TabNetRegressor

__version__ = version("pytorch-tabnet2")
