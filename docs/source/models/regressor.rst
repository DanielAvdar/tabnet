TabNet Regressor
================

.. automodule:: pytorch_tabnet
    :members: TabNetRegressor
    :undoc-members:
    :show-inheritance:
    :no-index:

Example
-------

.. code-block:: python

    from pytorch_tabnet import TabNetRegressor
    import numpy as np
    X = np.random.rand(100, 10)
    y = np.random.rand(100, 1)
    reg = TabNetRegressor()
    reg.fit(X_train=X, y_train=y)
    preds = reg.predict(X)
