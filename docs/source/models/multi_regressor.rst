TabNet Multi-Task Regressor
===========================

.. automodule:: pytorch_tabnet
    :members: MultiTabNetRegressor
    :undoc-members:
    :show-inheritance:
    :no-index:

Example
-------

.. code-block:: python


    from pytorch_tabnet import MultiTabNetRegressor
    import numpy as np
    X = np.random.rand(100, 10)
    y = np.random.rand(100, 3)  # 3 regression targets
    reg = MultiTabNetRegressor()
    reg.fit(X_train=X, y_train=y)
    preds = reg.predict(X)
