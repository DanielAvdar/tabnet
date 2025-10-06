"""Utility functions for TabNet package."""

from .device import define_device as define_device
from .dimension import infer_multitask_output as infer_multitask_output
from .dimension import infer_output_dim as infer_output_dim
from .matrices import _create_explain_matrix as create_explain_matrix  # noqa
from .matrices import create_group_matrix as create_group_matrix
from .serialization import ComplexEncoder as ComplexEncoder

# Gmail API utilities (optional import - requires google-api-python-client)
try:
    from .gmail_api import (  # noqa: F401
        GmailAPIClient,
        GmailBatchError,
        batch_operation_with_retry,
        create_credentials,
        get_message_headers,
    )

    __all_gmail__ = ["GmailAPIClient", "GmailBatchError", "create_credentials", "get_message_headers", "batch_operation_with_retry"]
except ImportError:
    __all_gmail__ = []

__all__ = [
    "define_device",
    "infer_multitask_output",
    "infer_output_dim",
    "create_explain_matrix",
    "create_group_matrix",
    "ComplexEncoder",
] + __all_gmail__
