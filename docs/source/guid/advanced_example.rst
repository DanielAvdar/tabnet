Advanced Example Guide
======================

This section provides advanced usage examples for eh-pytorch-tabnet.

.. contents:: Table of Contents
   :depth: 1

Advanced Example 1: Custom Loss Function
----------------------------------------

This guide demonstrates how to use a custom loss function with TabNet.

.. code-block:: python

   import numpy as np
   from pytorch_tabnet.tab_model import TabNetRegressor
   import torch.nn as nn
   import torch

   # Generate dummy data
   X_train = np.random.rand(100, 10).astype(np.float32)
   y_train = np.random.rand(100).astype(np.float32).reshape(-1, 1)

   def custom_loss(y_true, y_pred):
       return nn.functional.mse_loss(y_pred, y_true) + 0.1 * torch.mean(torch.abs(y_pred))

   model = TabNetRegressor()
   model.fit(X_train, y_train, loss_fn=custom_loss)

More advanced guides coming soon!
