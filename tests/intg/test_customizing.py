


def test_customizing():
    from pytorch_tabnet.tab_model import TabNetClassifier

    import torch
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import roc_auc_score

    import pandas as pd
    import numpy as np
    np.random.seed(0)

    import os
    import wget
    from pathlib import Path

    from matplotlib import pyplot as plt
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    dataset_name = 'census-income'
    out = Path(os.getcwd() + '/data/' + dataset_name + '.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print("File already exists.")
    else:
        print("Downloading file...")
        wget.download(url, out.as_posix())
    train = pd.read_csv(out)
    target = ' <=50K'
    if "Set" not in train.columns:
        train["Set"] = np.random.choice(["train", "valid", "test"], p=[.8, .1, .1], size=(train.shape[0],))

    train_indices = train[train.Set == "train"].index
    valid_indices = train[train.Set == "valid"].index
    test_indices = train[train.Set == "test"].index
    nunique = train.nunique()
    types = train.dtypes

    categorical_columns = []
    categorical_dims = {}
    for col in train.columns:
        if types[col] == 'object' or nunique[col] < 200:

            l_enc = LabelEncoder()
            train[col] = train[col].fillna("VV_likely")
            train[col] = l_enc.fit_transform(train[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            train.fillna(train.loc[train_indices, col].mean(), inplace=True)
    unused_feat = ['Set']

    features = [col for col in train.columns if col not in unused_feat + [target]]

    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]

    cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    max_epochs =  4
    batch_size = 1024
    clf = TabNetClassifier(cat_idxs=cat_idxs,
                           cat_dims=cat_dims,
                           cat_emb_dim=1,
                           optimizer_fn=torch.optim.Adam,  # Any optimizer works here
                           optimizer_params=dict(lr=2e-2),
                           scheduler_fn=torch.optim.lr_scheduler.OneCycleLR,
                           scheduler_params={"is_batch_level": True,
                                             "max_lr": 5e-2,
                                             "steps_per_epoch": int(train.shape[0] / batch_size) + 1,
                                             "epochs": max_epochs
                                             },
                           mask_type='entmax',  # "sparsemax",
                           )
    X_train = train[features].values[train_indices]
    y_train = train[target].values[train_indices]

    X_valid = train[features].values[valid_indices]
    y_valid = train[target].values[valid_indices]

    X_test = train[features].values[test_indices]
    y_test = train[target].values[test_indices]

    def my_loss_fn(y_pred, y_true):
        softmax_pred = torch.nn.Softmax(dim=-1)(y_pred)
        logloss = (1 - y_true) * torch.log(softmax_pred[:, 0])
        logloss += y_true * torch.log(softmax_pred[:, 1])
        return -torch.mean(logloss)

    from pytorch_tabnet.metrics import Metric
    class my_metric(Metric):
    
        def __init__(self):
            self._name = "custom"
            self._maximize = True
    
        def __call__(self, y_true, y_score):
            return 2 * roc_auc_score(y_true, y_score[:, 1])

    # %%
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'val'],
        eval_metric=["auc", my_metric],
        max_epochs=max_epochs, patience=0,
        batch_size=batch_size,
        virtual_batch_size=128,
        num_workers=0,
        weights=1,
        drop_last=False,
        loss_fn=my_loss_fn
    )

    plt.plot(clf.history['loss'])
    plt.plot(clf.history['train_auc'])
    plt.plot(clf.history['val_auc'])
    plt.plot(clf.history['lr'])
    preds = clf.predict_proba(X_test)
    test_auc = roc_auc_score(y_score=preds[:, 1], y_true=y_test)

    preds_valid = clf.predict_proba(X_valid)
    valid_auc = roc_auc_score(y_score=preds_valid[:, 1], y_true=y_valid)

    print(f"FINAL VALID SCORE FOR {dataset_name} : {clf.history['val_auc'][-1]}")
    print(f"FINAL TEST SCORE FOR {dataset_name} : {test_auc}")
    assert test_auc > 0.9
    # %%
    # check that last epoch's weight are used
    assert np.isclose(valid_auc, clf.history['val_auc'][-1], atol=1e-6)
    saving_path_name = "./tabnet_model_test_1"
    saved_filepath = clf.save_model(saving_path_name)
    loaded_clf = TabNetClassifier()
    loaded_clf.load_model(saved_filepath)
    loaded_preds = loaded_clf.predict_proba(X_test)
    loaded_test_auc = roc_auc_score(y_score=loaded_preds[:, 1], y_true=y_test)

    print(f"FINAL TEST SCORE FOR {dataset_name} : {loaded_test_auc}")
    assert (test_auc == loaded_test_auc)

    clf.feature_importances_

    explain_matrix, masks = clf.explain(X_test)
    fig, axs = plt.subplots(1, 3, figsize=(20, 20))

    for i in range(3):
        axs[i].imshow(masks[i][:50])
        axs[i].set_title(f"mask {i}")

