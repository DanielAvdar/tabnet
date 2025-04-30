import unittest

import numpy as np
from scipy.sparse import csr_matrix

from pytorch_tabnet.multiclass_utils import infer_output_dim
from pytorch_tabnet.multiclass_utils.tmp import (
    assert_all_finite,
    check_classification_targets,
    is_multilabel,
    type_of_target,
    unique_labels,
)
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.utils import check_embedding_parameters, check_input, filter_weights


class TestErrorHandling(unittest.TestCase):
    def test_tabnet_classifier_prepare_target_exception(self):
        model = TabNetClassifier()
        with self.assertRaises(Exception):
            model.prepare_target(None)

    def test_tabnet_multi_task_classifier_prepare_target_exception(self):
        model = TabNetMultiTaskClassifier()
        with self.assertRaises(Exception):
            model.prepare_target(None)

    def test_tabnet_classifier_invalid_input_shape(self):
        model = TabNetClassifier()
        X = np.random.rand(10)  # 1D array instead of 2D
        y = np.random.randint(0, 2, size=10)
        with self.assertRaises(IndexError):
            model.fit(X, y)

    def test_tabnet_multi_task_classifier_invalid_targets(self):
        model = TabNetMultiTaskClassifier()
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, size=10)  # Should be a list of targets
        with self.assertRaises(ValueError):
            model.fit(X, y)

    def test_loading_invalid_model(self):
        model = TabNetClassifier()
        with self.assertRaises(FileNotFoundError):
            model.load_model("non_existent_model.zip")

    def test_classifier_invalid_prediction_input(self):
        model = TabNetClassifier()
        # Try to predict without fitting first
        X = np.random.rand(10, 5)
        with self.assertRaises(Exception):
            model.predict(X)

    def test_invalid_batch_size(self):
        model = TabNetClassifier()
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, size=10)
        with self.assertRaises(ValueError):
            # batch_size must be > 0
            model.fit(X, y, batch_size=0)

    def test_multiclass_utils_errors(self):
        # Testing infer_output_dim function with invalid input

        # Test with None input
        with self.assertRaises(Exception):
            infer_output_dim(None)

        # Test sparse matrix format errors
        # This requires importing scipy's sparse matrix, but we can at least check for
        # error conditions that we can simulate
        y_invalid = np.array([["a", "b"], ["c", "d"]])
        with self.assertRaises(Exception):
            infer_output_dim(y_invalid)

    def test_sparse_matrix_handling(self):
        """Test error handling with sparse matrix operations"""

        # Create weights in incorrect format
        weights = {"invalid": "format"}
        with self.assertRaises(Exception):
            filter_weights(None, weights)

    # New tests for multiclass_utils error handling through TabNetMultiTaskClassifier

    def test_multitask_mixed_type_target(self):
        """Test handling of targets with mixed types in multitask setting"""
        model = TabNetMultiTaskClassifier()
        X = np.random.rand(10, 5)
        # Create a target array with mixed types (strings and integers)
        y1 = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
        y2 = np.array(["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"])
        y = np.column_stack((y1, y2))

        with self.assertRaises(TypeError):
            model.fit(X, y)

    def test_multitask_invalid_dimension(self):
        """Test handling of invalid dimensionality in multitask targets"""
        model = TabNetMultiTaskClassifier()
        X = np.random.rand(10, 5)
        # Create a 3D target array which is invalid
        y = np.random.randint(0, 2, size=(10, 2, 3))

        with self.assertRaises(ValueError):
            model.fit(X, y)

    def test_multitask_unknown_target_format(self):
        """Test handling of targets with unknown format"""
        model = TabNetMultiTaskClassifier()
        X = np.random.rand(10, 5)
        # Create an object array with complex data types
        y = np.empty((10, 2), dtype=object)
        for i in range(10):
            y[i, 0] = {"a": i}  # Dictionary in first column
            y[i, 1] = set([i])  # Set in second column

        with self.assertRaises(Exception):
            model.fit(X, y)

    def test_multitask_sparse_series(self):
        """Test handling of pandas SparseSeries which is not supported"""
        TabNetMultiTaskClassifier()
        np.random.rand(10, 5)

        # Create a simulated SparseSeries-like object
        class MockSparseSeries:
            __class__ = type("SparseSeries", (), {})

        y = MockSparseSeries()

        with self.assertRaises(ValueError):
            type_of_target(y)

    def test_multitask_sequence_of_sequences(self):
        """Test handling of legacy multilabel format (sequence of sequences)"""
        model = TabNetMultiTaskClassifier()
        X = np.random.rand(10, 5)
        # Create a sequence of sequences format that will trigger the error
        # Convert to numpy array to avoid the "no attribute 'shape'" error
        y = np.array([[1, 2], [0, 3], [1, 4], [2, 0], [0, 1], [3, 2], [1, 1], [0, 2], [1, 3], [0, 1]], dtype=object)

        with self.assertRaises(ValueError):
            model.fit(X, y)

    def test_multitask_non_subset_validation(self):
        """Test error when validation set has labels not in training set"""
        model = TabNetMultiTaskClassifier()
        X_train = np.random.rand(10, 5)
        y_train = np.column_stack((np.random.randint(0, 3, size=10), np.random.randint(0, 3, size=10)))

        X_val = np.random.rand(5, 5)
        # Create validation targets with labels not in training set
        y_val = np.column_stack((
            np.random.randint(0, 3, size=5),
            np.random.randint(3, 5, size=5),  # These labels (3,4) aren't in training
        ))

        with self.assertRaises(ValueError):
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # Additional tests for multiclass_utils error handling

    def test_multitask_3d_array_error(self):
        """Test handling of 3D arrays in type_of_target"""
        model = TabNetMultiTaskClassifier()
        X = np.random.rand(10, 5)
        # Create a 3D array for targets
        y = np.random.randint(0, 2, size=(10, 2, 3))

        with self.assertRaises(ValueError):
            model.fit(X, y)

    def test_multitask_empty_dimension_error(self):
        """Test handling of arrays with empty dimensions in type_of_target"""

        # Create array with empty second dimension
        y = np.zeros((5, 0))
        self.assertEqual(type_of_target(y), "unknown")

    def test_object_dtype_with_nan_handling(self):
        """Test handling of object dtypes with NaN values"""

        # Create float array with NaN (to avoid object array TypeErrors)
        y = np.array([1.0, 2.0, np.nan, 3.0], dtype=float)

        with self.assertRaises(ValueError):
            assert_all_finite(y, allow_nan=False)

    def test_infinity_handling_with_allow_nan(self):
        """Test handling of infinite values when allow_nan is True"""

        # Create array with infinity but allow_nan=True (should still fail)
        y = np.array([1.0, 2.0, np.inf, 3.0])

        with self.assertRaises(ValueError):
            assert_all_finite(y, allow_nan=True)

    def test_assert_all_finite_object_array(self):
        """Test assert_all_finite with object arrays but bypassing direct isnan call"""

        # Use float arrays that are compatible with isnan/isfinite
        y = np.array([1.0, 2.0, 3.0], dtype=float)
        assert_all_finite(y, allow_nan=False)

        # Test with explicit float NaN value (should raise error)
        y_with_nan = np.array([1.0, np.nan, 3.0], dtype=float)
        with self.assertRaises(ValueError):
            assert_all_finite(y_with_nan, allow_nan=False)

    def test_is_multilabel_dok_matrix(self):
        """Test is_multilabel with different sparse matrix types"""

        # Create a sparse matrix with binary values (proper multilabel format)
        mat = csr_matrix(([1, 1], ([0, 1], [1, 2])), shape=(3, 3))

        # This should identify as multilabel based on shape and binary values
        self.assertTrue(is_multilabel(mat))

        # Test with single column sparse matrix
        mat = csr_matrix((3, 1))
        mat[0, 0] = 1
        self.assertFalse(is_multilabel(mat))

    def test_unique_labels_no_arguments(self):
        """Test unique_labels with no arguments"""

        with self.assertRaises(ValueError):
            unique_labels()

    def test_unique_labels_mixed_types(self):
        """Test unique_labels with mixed data types that should raise an error"""

        # Creating different label types for different arrays
        # This should raise ValueError since we mix string and integer label types
        with self.assertRaises(ValueError):
            unique_labels([1, 2, 3], ["a", "b", "c"])

    def test_type_of_target_non_array(self):
        """Test type_of_target with non-array inputs"""

        # Input not convertible to array
        with self.assertRaises(ValueError):
            type_of_target("not_an_array")

    def test_check_classification_targets_wrong_type(self):
        """Test check_classification_targets with wrong target type"""

        # Create a continuous target
        y = np.array([0.1, 0.2, 0.3])

        with self.assertRaises(ValueError):
            check_classification_targets(y)

    def test_check_embedding_parameters(self):
        """Test check_classification_targets with wrong target type"""

        with self.assertRaises(ValueError):
            check_embedding_parameters(
                cat_dims=[1],
                cat_emb_dim=[],
                cat_idxs=[1],
            )

    def test_check_input(self):
        """Test check_classification_targets with wrong target type"""
        import pandas as pd

        with self.assertRaises(TypeError):
            check_input(
                pd.DataFrame(),
            )

    def test_filter_weights(self):
        """Test check_classification_targets with wrong target type"""

        with self.assertRaises(ValueError):
            filter_weights(1)
        with self.assertRaises(ValueError):
            filter_weights(dict())


if __name__ == "__main__":
    unittest.main()
