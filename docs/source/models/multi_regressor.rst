.. _multi_regressor:

TabNet Multi-Task Regressor
===========================

.. automodule:: pytorch_tabnet
    :members: TabNetMultiTaskRegressor
    :undoc-members:
    :show-inheritance:
    :no-index:

Example
-------

.. code-block:: python


    from pytorch_tabnet import TabNetMultiTaskRegressor
    import numpy as np
    X = np.random.rand(100, 10)
    y = np.random.rand(100, 3)  # 3 regression targets
    reg = TabNetMultiTaskRegressor()
    reg.fit(X_train=X, y_train=y)
    preds = reg.predict(X)
