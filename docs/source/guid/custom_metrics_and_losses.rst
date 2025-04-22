Custom Metrics and Losses Guide
==============================

This guide demonstrates how to use custom evaluation metrics and custom loss functions with TabNet. Each example is standalone.

Custom Evaluation Metric Example
-------------------------------

.. code-block:: python

   import numpy as np
   from pytorch_tabnet.tab_model import TabNetClassifier
   from pytorch_tabnet.metrics import Metric
   from sklearn.metrics import roc_auc_score

   class Gini(Metric):
       def __init__(self):
           self._name = "gini"
           self._maximize = True

       def __call__(self, y_true, y_score, weights=None):
           auc = roc_auc_score(y_true, y_score[:, 1])
           return max(2*auc - 1, 0.)

   # Generate dummy data
   X_train = np.random.rand(100, 10)
   y_train = np.random.randint(0, 2, 100)
   X_valid = np.random.rand(20, 10)
   y_valid = np.random.randint(0, 2, 20)

   clf = TabNetClassifier()
   clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric=[Gini])

Custom Loss Function Example
---------------------------

.. code-block:: python

   import numpy as np
   import torch
   import torch.nn as nn
   from pytorch_tabnet.tab_model import TabNetRegressor

   # Generate dummy data
   X_train = np.random.rand(100, 10).astype(np.float32)
   y_train = np.random.rand(100).astype(np.float32).reshape(-1, 1)
   X_valid = np.random.rand(20, 10).astype(np.float32)
   y_valid = np.random.rand(20).astype(np.float32).reshape(-1, 1)

   import torch
   def custom_loss(y_true, y_pred):
       return nn.functional.mse_loss(y_pred, y_true) + 0.1 * torch.mean(torch.abs(y_pred))

   reg = TabNetRegressor()
   reg.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], loss_fn=custom_loss)
