.. _pretrainer:

TabNet Pretrainer
=================

.. automodule:: pytorch_tabnet
    :members: TabNetPretrainer
    :undoc-members:
    :show-inheritance:

Example
-------

.. code-block:: python

    from pytorch_tabnet.pretraining import TabNetPretrainer
    import numpy as np
    X = np.random.rand(100, 10)
    pretrainer = TabNetPretrainer()
    pretrainer.fit(X_train=X)
