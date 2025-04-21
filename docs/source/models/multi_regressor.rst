.. _multi_regressor:

TabNet Multi-Task Regressor
==========================

.. automodule:: pytorch_tabnet.tab_model
    :members: MultiTabNetRegressor
    :undoc-members:
    :show-inheritance:

Example
-------

.. code-block:: python

    from pytorch_tabnet.tab_model import MultiTabNetRegressor
    import numpy as np
    X = np.random.rand(100, 10)
    y = np.random.rand(100, 3)  # 3 regression targets
    reg = MultiTabNetRegressor()
    reg.fit(X_train=X, y_train=y)
    preds = reg.predict(X)
