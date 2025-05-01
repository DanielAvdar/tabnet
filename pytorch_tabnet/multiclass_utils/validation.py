"""Import from utils module for backward compatibility."""

from pytorch_tabnet.utils.multiclass_validation import check_unique_type as _original_check_unique_type, assert_all_finite  # noqa
from pytorch_tabnet.utils.validation_multi import check_unique_type, check_output_dim  # noqa
