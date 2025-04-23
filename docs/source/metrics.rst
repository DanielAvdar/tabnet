Metrics
=======

This section documents the metrics available in ``pytorch_tabnet.metrics``. These metrics are used for evaluating model performance in classification, regression, and unsupervised tasks.

.. contents:: Table of Contents
   :depth: 1

Classification Metrics
---------------------

**Accuracy**
^^^^^^^^^^^^
Measures the proportion of correct predictions among the total number of cases.

.. math::
   \mathrm{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}

*Example:*

.. code-block:: python

   import torch
   from pytorch_tabnet.metrics import Accuracy
   metric = Accuracy()
   y_true = torch.tensor([0, 1, 1, 0])
   y_pred = torch.tensor([0, 1, 0, 0])
   print(metric(y_true, y_pred))  # Output: 0.75

**AUC (Area Under the ROC Curve)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Represents the probability that a classifier ranks a random positive instance higher than a random negative one.

.. math::
   \mathrm{AUC} = \int_{0}^{1} \mathrm{TPR}(\mathrm{FPR}^{-1}(x)) \, dx

*Example:*

.. code-block:: python

   import torch
   from pytorch_tabnet.metrics import AUC
   metric = AUC()
   y_true = torch.tensor([0, 1, 1, 0])
   y_score = torch.tensor([0.1, 0.8, 0.7, 0.2])
   print(metric(y_true, y_score))  # Output: 1.0

**Balanced Accuracy**
^^^^^^^^^^^^^^^^^^^^^
Computes the average recall obtained on each class, useful for imbalanced datasets.

.. math::
   \mathrm{Balanced\ Accuracy} = \frac{1}{2} \left( \frac{TP}{TP+FN} + \frac{TN}{TN+FP} \right)

*Example:*

.. code-block:: python

   import torch
   from pytorch_tabnet.metrics import BalancedAccuracy
   metric = BalancedAccuracy()
   y_true = torch.tensor([0, 1, 1, 0])
   y_pred = torch.tensor([0, 1, 0, 0])
   print(metric(y_true, y_pred))  # Output: 0.75

**Log Loss (Cross-Entropy Loss)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Measures the performance of a classification model where the prediction is a probability value between 0 and 1.

.. math::
   \mathrm{LogLoss} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]

*Example:*

.. code-block:: python

   import torch
   from pytorch_tabnet.metrics import LogLoss
   metric = LogLoss()
   y_true = torch.tensor([0, 1])
   y_pred = torch.tensor([0.1, 0.9])
   print(metric(y_true, y_pred))  # Output: 0.105...

Regression Metrics
------------------

**Mean Absolute Error (MAE)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Measures the average magnitude of the errors in a set of predictions, without considering their direction.

.. math::
   \mathrm{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|

*Example:*

.. code-block:: python

   import torch
   from pytorch_tabnet.metrics import MAE
   metric = MAE()
   y_true = torch.tensor([3, -0.5, 2, 7])
   y_pred = torch.tensor([2.5, 0.0, 2, 8])
   print(metric(y_true, y_pred))  # Output: 0.5

**Mean Squared Error (MSE)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Measures the average of the squares of the errors between actual and predicted values.

.. math::
   \mathrm{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2

*Example:*

.. code-block:: python

   import torch
   from pytorch_tabnet.metrics import MSE
   metric = MSE()
   y_true = torch.tensor([3, -0.5, 2, 7])
   y_pred = torch.tensor([2.5, 0.0, 2, 8])
   print(metric(y_true, y_pred))  # Output: 0.375

**Root Mean Squared Error (RMSE)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The square root of the mean squared error, providing error in the same units as the target variable.

.. math::
   \mathrm{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}

*Example:*

.. code-block:: python

   import torch
   from pytorch_tabnet.metrics import RMSE
   metric = RMSE()
   y_true = torch.tensor([3, -0.5, 2, 7])
   y_pred = torch.tensor([2.5, 0.0, 2, 8])
   print(metric(y_true, y_pred))  # Output: 0.612...

**Root Mean Squared Logarithmic Error (RMSLE)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Measures the ratio between the true and predicted values, less sensitive to large errors when both values are large.

.. math::
   \mathrm{RMSLE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( \log(y_i + 1) - \log(\hat{y}_i + 1) \right)^2}

*Example:*

.. code-block:: python

   import torch
   from pytorch_tabnet.metrics import RMSLE
   metric = RMSLE()
   y_true = torch.tensor([3, 5, 2.5, 7])
   y_pred = torch.tensor([2.5, 5, 4, 8])
   print(metric(y_true, y_pred))  # Output: 0.120...

Unsupervised Metrics
--------------------

**Unsupervised Loss**
^^^^^^^^^^^^^^^^^^^^^
Used for unsupervised pretraining, typically measures reconstruction error.

*Example:*

.. code-block:: python

   import torch
   from pytorch_tabnet.metrics import UnsupervisedLoss
   metric = UnsupervisedLoss()
   # Example usage depends on the unsupervised task

**Unsupervised Metrics**
^^^^^^^^^^^^^^^^^^^^^^^^
Additional metrics for unsupervised learning tasks.

*Example:*

.. code-block:: python

   import torch
   from pytorch_tabnet.metrics import UnsupervisedMetric
   metric = UnsupervisedMetric()
   # Example usage depends on the unsupervised task

Base Metrics
------------

**Base Metrics**
^^^^^^^^^^^^^^^^
Contains base classes and utilities for defining custom metrics.

*Example:*

.. code-block:: python

   import torch
   from pytorch_tabnet.metrics import Metric
   # See source for custom metric implementation

See the source code in ``pytorch_tabnet/metrics/`` for implementation details and how to use these metrics in your models.
