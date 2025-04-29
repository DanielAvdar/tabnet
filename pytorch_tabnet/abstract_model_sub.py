from dataclasses import dataclass

from pytorch_tabnet.abstract_model import TabModel


@dataclass
class TabSupervisedModel(TabModel):
    """Abstract base class for TabNet models."""
