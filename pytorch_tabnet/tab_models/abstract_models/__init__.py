"""Abstract model definitions for TabNet."""

from .abs_model import TabModel
from .supervised_model import TabSupervisedModel

__all__ = ["TabSupervisedModel", "TabModel"]
