"""pytorch_tabnet package initialization."""

from importlib.metadata import version, PackageNotFoundError

from .tab_models.multitask import TabNetMultiTaskClassifier as TabNetMultiTaskClassifier
from .tab_models.pretraining import TabNetPretrainer as TabNetPretrainer
from .tab_models.tab_class import TabNetClassifier as TabNetClassifier
from .tab_models.tab_reg import MultiTabNetRegressor as MultiTabNetRegressor
from .tab_models.tab_reg import TabNetRegressor as TabNetRegressor

try:
    __version__ = version("pytorch-tabnet2")
except PackageNotFoundError:
    # Package is not installed, set a development version
    __version__ = "0.0.0.dev"
