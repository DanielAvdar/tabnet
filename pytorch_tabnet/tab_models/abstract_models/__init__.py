"""Abstract model definitions for TabNet."""

from .base_model import TabModel
from .supervised_model import TabSupervisedModel

__all__ = ["TabModel", "TabSupervisedModel"]
