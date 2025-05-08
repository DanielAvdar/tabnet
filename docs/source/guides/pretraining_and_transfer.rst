Pretraining and Transfer Learning
=================================

This guide demonstrates how to use TabNet's semi-supervised pretraining and transfer learning. Each example is standalone.

Pretraining Example
-------------------

.. code-block:: python

   import numpy as np
   import torch
   from pytorch_tabnet import TabNetPretrainer

   # Generate dummy data
   X_train = np.random.rand(100, 10).astype(np.float32)
   X_valid = np.random.rand(20, 10).astype(np.float32)

   unsupervised_model = TabNetPretrainer(
       optimizer_fn=torch.optim.Adam,
       optimizer_params=dict(lr=2e-2),
       mask_type='entmax'  # or 'sparsemax'
   )
   unsupervised_model.fit(
       X_train=X_train,
       eval_set=[X_valid],
       pretraining_ratio=0.8,
   )

Transfer Learning Example
-------------------------

.. code-block:: python

   import numpy as np
   import torch
   from pytorch_tabnet import TabNetClassifier
   from pytorch_tabnet import TabNetPretrainer

   # Generate dummy data
   X_train = np.random.rand(100, 10).astype(np.float32)
   y_train = np.random.randint(0, 2, 100)
   X_valid = np.random.rand(20, 10).astype(np.float32)
   y_valid = np.random.randint(0, 2, 20)

   # Pretrain
   unsupervised_model = TabNetPretrainer(
       optimizer_fn=torch.optim.Adam,
       optimizer_params=dict(lr=2e-2),
       mask_type='entmax'
   )
   unsupervised_model.fit(
       X_train=X_train,
       eval_set=[X_valid],
       pretraining_ratio=0.8,
   )

   # Fine-tune
   clf = TabNetClassifier(
       optimizer_fn=torch.optim.Adam,
       optimizer_params=dict(lr=2e-2),
       scheduler_params={"step_size": 10, "gamma": 0.9},
       scheduler_fn=torch.optim.lr_scheduler.StepLR,
       mask_type='sparsemax'
   )
   clf.fit(
       X_train=X_train, y_train=y_train,
       eval_set=[(X_train, y_train), (X_valid, y_valid)],
       eval_name=['train', 'valid'],
       eval_metric=['auc'],
       from_unsupervised=unsupervised_model
   )
