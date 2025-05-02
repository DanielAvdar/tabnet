"""Utility functions for TabNet package."""

from .device import define_device as define_device
from .dimension import infer_multitask_output as infer_multitask_output
from .dimension import infer_output_dim as infer_output_dim
from .matrices import _create_explain_matrix as create_explain_matrix  # noqa
from .matrices import create_group_matrix as create_group_matrix
from .multiclass_validation import check_output_dim as check_output_dim
from .serialization import ComplexEncoder as ComplexEncoder
from .validation import check_embedding_parameters as check_embedding_parameters
from .validation import check_input as check_input
from .validation import filter_weights as filter_weights
from .validation import validate_eval_set as validate_eval_set
