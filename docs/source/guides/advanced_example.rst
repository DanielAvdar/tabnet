Categorical Embedding Example Guide
===================================

This section provides examples for using categorical embeddings with eh-pytorch-tabnet.

.. contents:: Table of Contents
   :depth: 1

Categorical Embedding Example: Classification
---------------------------------------------

This guide demonstrates how to use categorical features with embeddings in TabNet for a classification task.

.. code-block:: python

   import numpy as np
   from pytorch_tabnet.tab_model import TabNetClassifier

   # Generate dummy data
   X_train = np.random.randint(0, 5, size=(100, 3))  # 3 categorical features with 5 categories each
   X_train = np.concatenate([
       X_train,
       np.random.rand(100, 7)  # 7 continuous features
   ], axis=1).astype(np.float32)
   y_train = np.random.randint(0, 2, size=(100,))

   # Specify categorical feature indices and their dimensions
   cat_idxs = [0, 1, 2]  # indices of categorical columns
   cat_dims = [5, 5, 5]  # number of unique values for each categorical column

   model = TabNetClassifier(cat_idxs=cat_idxs, cat_dims=cat_dims)
   model.fit(X_train, y_train)

Categorical Embedding Example: Regression
-----------------------------------------

This guide demonstrates how to use categorical features with embeddings in TabNet for a regression task.

.. code-block:: python

   import numpy as np
   from pytorch_tabnet.tab_model import TabNetRegressor

   # Generate dummy data
   X_train = np.random.randint(0, 4, size=(100, 2))  # 2 categorical features with 4 categories each
   X_train = np.concatenate([
       X_train,
       np.random.rand(100, 8)  # 8 continuous features
   ], axis=1).astype(np.float32)
   y_train = np.random.rand(100)

   # Specify categorical feature indices and their dimensions
   cat_idxs = [0, 1]  # indices of categorical columns
   cat_dims = [4, 4]  # number of unique values for each categorical column

   model = TabNetRegressor(cat_idxs=cat_idxs, cat_dims=cat_dims)
   model.fit(X_train, y_train)

.. note::
   When using categorical features, ensure that the categorical columns are integer-encoded (0 to N-1 for N categories).

More categorical embedding guides coming soon!
