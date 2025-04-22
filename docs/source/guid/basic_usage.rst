Basic Usage Guide
=================

This guide demonstrates basic usage of TabNet for classification, regression, and multi-task problems. Each example is standalone and can be run independently.

Classification Example
----------------------

.. code-block:: python

   import numpy as np
   from pytorch_tabnet.tab_model import TabNetClassifier

   # Generate dummy data
   X_train = np.random.rand(100, 10)
   y_train = np.random.randint(0, 2, 100)
   X_valid = np.random.rand(20, 10)
   y_valid = np.random.randint(0, 2, 20)
   X_test = np.random.rand(10, 10)

   clf = TabNetClassifier()
   clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
   preds = clf.predict(X_test)
   print('Predictions:', preds)

Regression Example
------------------

.. code-block:: python

   import numpy as np
   from pytorch_tabnet.tab_model import TabNetRegressor

   # Generate dummy data
   X_train = np.random.rand(100, 10)
   y_train = np.random.rand(100).reshape(-1, 1)
   X_valid = np.random.rand(20, 10)
   y_valid = np.random.rand(20).reshape(-1, 1)
   X_test = np.random.rand(10, 10)

   reg = TabNetRegressor()
   reg.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
   preds = reg.predict(X_test)
   print('Predictions:', preds)

Multi-task Classification Example
---------------------------------

.. code-block:: python

   import numpy as np
   from pytorch_tabnet.multitask import TabNetMultiTaskClassifier

   # Generate dummy data
   X_train = np.random.rand(100, 10)
   y_train = np.random.randint(0, 2, (100, 3))  # 3 tasks
   X_valid = np.random.rand(20, 10)
   y_valid = np.random.randint(0, 2, (20, 3))
   X_test = np.random.rand(10, 10)

   clf = TabNetMultiTaskClassifier()
   clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
   preds = clf.predict(X_test)
   print('Predictions:', preds)
