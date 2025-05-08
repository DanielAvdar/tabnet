.. _classifier:

TabNet Classifier
=================

.. automodule:: pytorch_tabnet
    :members: TabNetClassifier
    :undoc-members:
    :show-inheritance:
    :no-index:

Example
-------

.. code-block:: python

    from pytorch_tabnet import TabNetClassifier
    import numpy as np
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, size=(100,))
    clf = TabNetClassifier()
    clf.fit(X_train=X, y_train=y)
    preds = clf.predict(X)
