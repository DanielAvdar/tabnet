Regression Metrics
==================

This section documents the regression metrics available in ``pytorch_tabnet.metrics``.

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
