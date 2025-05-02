import pytorch_tabnet.error_handlers


def test_error_handlers_package_import():
    """Test that the error_handlers package can be imported."""
    assert pytorch_tabnet.error_handlers is not None
