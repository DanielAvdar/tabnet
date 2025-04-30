Data Augmentation and Callbacks
=====================================

This guide demonstrates how to use data augmentation and callbacks with TabNet. Each example is standalone.

.. warning::
   **Deprecation Notice:** The ``augmentations`` parameter is deprecated and will be removed in a future version.

Data Augmentation Example
------------------------

.. code-block:: python

   import numpy as np
   from pytorch_tabnet.tab_model import TabNetClassifier
   from pytorch_tabnet.augmentations import ClassificationSMOTE

   # Generate dummy data
   X_train = np.random.rand(100, 10)
   y_train = np.random.randint(0, 2, 100)
   X_valid = np.random.rand(20, 10)
   y_valid = np.random.randint(0, 2, 20)

   # Note: This approach is deprecated and will be removed in a future version
   aug = ClassificationSMOTE()
   clf = TabNetClassifier()
   clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], augmentations=aug)

Custom Callback Example
----------------------

.. code-block:: python

   import numpy as np
   from pytorch_tabnet.tab_model import TabNetClassifier
   from pytorch_tabnet.callbacks import Callback

   # Generate dummy data
   X_train = np.random.rand(100, 10)
   y_train = np.random.randint(0, 2, 100)
   X_valid = np.random.rand(20, 10)
   y_valid = np.random.randint(0, 2, 20)

   class PrintEpochCallback(Callback):
       def on_epoch_end(self, epoch, logs=None):
           print(f"Epoch {epoch} ended.")

   clf = TabNetClassifier()
   clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[PrintEpochCallback()])
