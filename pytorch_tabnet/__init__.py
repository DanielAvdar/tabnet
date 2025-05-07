"""pytorch_tabnet package initialization."""

from importlib.metadata import version

from .tab_models.multitask import TabNetMultiTaskClassifier as TabNetMultiTaskClassifier
from .tab_models.pretraining import TabNetPretrainer as TabNetPretrainer
from .tab_models.tab_model import MultiTabNetRegressor as MultiTabNetRegressor
from .tab_models.tab_model import TabNetClassifier as TabNetClassifier
from .tab_models.tab_model import TabNetRegressor as TabNetRegressor

__version__ = version("pytorch-tabnet2")
