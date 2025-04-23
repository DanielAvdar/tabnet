Classification Metrics
=====================

This section documents the classification metrics available in ``pytorch_tabnet.metrics``.

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
