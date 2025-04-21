TabNet Multi-Task Classifier
===========================

.. automodule:: pytorch_tabnet.multitask
    :members: TabNetMultiTaskClassifier
    :undoc-members:
    :show-inheritance:

Example
-------

.. code-block:: python

    from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
    import numpy as np
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, size=(100, 2))  # 2 classification tasks
    clf = TabNetMultiTaskClassifier()
    clf.fit(X_train=X, y_train=y)
    preds = clf.predict(X)
