Unsupervised Metrics
====================

This section documents the unsupervised metrics available in ``pytorch_tabnet.metrics``.

**Unsupervised Loss**
^^^^^^^^^^^^^^^^^^^^^
Used for unsupervised pretraining, typically measures reconstruction error.

*Example:*

.. code-block:: python

   import torch
   from pytorch_tabnet.metrics import UnsupervisedLoss
   y_pred = torch.randn(3, 5)
   embedded_x = torch.randn(3, 5)
   obf_vars = torch.randint(0, 2, (3, 5)).float()
   loss = UnsupervisedLoss(y_pred, embedded_x, obf_vars)
   print(loss)

**Unsupervised Metrics**
^^^^^^^^^^^^^^^^^^^^^^^^
Additional metrics for unsupervised learning tasks.

*Example:*

.. code-block:: python

   import torch
   from pytorch_tabnet.metrics import UnsupervisedMetric
   metric = UnsupervisedMetric()
   y_pred = torch.randn(3, 5)
   embedded_x = torch.randn(3, 5)
   obf_vars = torch.randint(0, 2, (3, 5)).float()
   score = metric(y_pred, embedded_x, obf_vars)
   print(score)

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
