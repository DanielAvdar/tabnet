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
