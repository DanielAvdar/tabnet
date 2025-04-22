Model Saving and Loading Guide
=============================

This guide demonstrates how to save and load TabNet models. Each example is standalone.

Saving and Loading a Classifier
------------------------------

.. code-block:: python

   import numpy as np
   from pytorch_tabnet.tab_model import TabNetClassifier

   # Generate dummy data
   X_train = np.random.rand(100, 10)
   y_train = np.random.randint(0, 2, 100)
   X_valid = np.random.rand(20, 10)
   y_valid = np.random.randint(0, 2, 20)

   clf = TabNetClassifier()
   clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

   # Save model
   saving_path_name = "./tabnet_model_test_1"
   saved_filepath = clf.save_model(saving_path_name)

   # Load model
   loaded_clf = TabNetClassifier()
   loaded_clf.load_model(saved_filepath)
   print("Model loaded successfully.")

Saving and Loading a Regressor
-----------------------------

.. code-block:: python

   import numpy as np
   from pytorch_tabnet.tab_model import TabNetRegressor

   # Generate dummy data
   X_train = np.random.rand(100, 10)
   y_train = np.random.rand(100)
   X_valid = np.random.rand(20, 10)
   y_valid = np.random.rand(20)

   reg = TabNetRegressor()
   reg.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

   # Save model
   saving_path_name = "./tabnet_model_test_2"
   saved_filepath = reg.save_model(saving_path_name)

   # Load model
   loaded_reg = TabNetRegressor()
   loaded_reg.load_model(saved_filepath)
   print("Regressor loaded successfully.")
