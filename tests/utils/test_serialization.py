import json

import numpy as np
import pytest

from pytorch_tabnet.utils.serialization import ComplexEncoder


class TestSerialization:
    """Tests for the serialization utility functions."""

    def test_complex_encoder_numpy_scalar(self):
        """Test encoding numpy scalar values."""
        data = {"value": np.int32(42)}
        result = json.dumps(data, cls=ComplexEncoder)
        assert json.loads(result) == {"value": 42}

    def test_complex_encoder_numpy_array(self):
        """Test encoding numpy arrays."""
        data = {"array": np.array([1, 2, 3])}
        result = json.dumps(data, cls=ComplexEncoder)
        assert json.loads(result) == {"array": [1, 2, 3]}

    def test_complex_encoder_default_behavior(self):
        """Test default behavior for non-numpy objects."""
        data = {"string": "test", "number": 42}
        result = json.dumps(data, cls=ComplexEncoder)
        assert json.loads(result) == data

    def test_complex_encoder_unsupported_type(self):
        """Test encoder with unsupported type."""

        # Create a custom class that can't be serialized
        class Unserializable:
            pass

        data = {"unserializable": Unserializable()}

        # The encoder should raise TypeError for unsupported types
        with pytest.raises(TypeError):
            json.dumps(data, cls=ComplexEncoder)
