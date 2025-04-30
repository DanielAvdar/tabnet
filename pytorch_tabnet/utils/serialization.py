"""Serialization utility functions for TabNet."""

import json

import numpy as np

# import json
#
# import numpy as np


# class ComplexEncoder(json.JSONEncoder):
class ComplexEncoder(json.JSONEncoder):
    """Custom JSON encoder for complex numbers and numpy data types."""

    def default(self, obj: object) -> object:
        """Convert numpy objects to lists for JSON serialization.

        Parameters
        ----------
        obj : object
            The object to encode.

        Returns
        -------
        object
            A JSON-serializable object (list, dict, etc.) or calls the base class default method.

        """
        if isinstance(obj, (np.generic, np.ndarray)):
            return obj.tolist()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
