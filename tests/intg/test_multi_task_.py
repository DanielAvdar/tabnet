










def test_multi_task():
    from pytorch_tabnet.multitask import TabNetMultiTaskClassifier

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
            print(col, train[col].nunique())
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

    clf = TabNetMultiTaskClassifier(cat_idxs=cat_idxs,
                                    cat_dims=cat_dims,
                                    cat_emb_dim=1,
                                    optimizer_fn=torch.optim.Adam,
                                    optimizer_params=dict(lr=2e-2),
                                    scheduler_params={"step_size": 50,
                                                      "gamma": 0.9},
                                    scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                    mask_type='entmax'
                                    )

    NB_TASKS = 5

    X_train = train[features].values[train_indices]
    y_train = train[target].values[train_indices].reshape(-1, 1)
    y_train = np.hstack([y_train] * NB_TASKS)
    y_train[:, -1] = np.random.randint(10, 15, y_train.shape[0]).astype(str)

    X_valid = train[features].values[valid_indices]
    y_valid = train[target].values[valid_indices].reshape(-1, 1)
    y_valid = np.hstack([y_valid] * NB_TASKS)
    y_valid[:, -1] = np.random.randint(10, 15, y_valid.shape[0]).astype(str)

    X_test = train[features].values[test_indices]
    y_test = train[target].values[test_indices].reshape(-1, 1)
    y_test = np.hstack([y_test] * NB_TASKS)
    y_test[:, -1] = np.random.randint(10, 15, y_test.shape[0]).astype(str)

    max_epochs =  5

    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'],
        max_epochs=max_epochs, patience=20,
        batch_size=1024,# virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        loss_fn=[torch.nn.functional.cross_entropy] * NB_TASKS
    )

    plt.plot(clf.history['loss'])

    plt.plot(clf.history['train_logloss'])
    plt.plot(clf.history['valid_logloss'])

    plt.plot(clf.history['lr'])

    preds = clf.predict_proba(X_test)
    test_aucs = [roc_auc_score(y_score=task_pred[:, 1], y_true=y_test[:, task_idx])
                 for task_idx, task_pred in enumerate(preds[:-1])]

    print(f"BEST VALID SCORE FOR {dataset_name} : {clf.best_cost}")
    print(f"FINAL AUC SCORES FOR {dataset_name} : {test_aucs}")

    predict_classes = clf.predict(X_test)

    clf.classes_

    predict_classes

    clf.target_mapper

    ensemble_auc = roc_auc_score(y_score=np.mean(np.vstack([task_pred[:, 1] for task_pred in preds]), axis=0),
                                 y_true=y_test[:, 0])

    ensemble_auc

    saving_path_name = "./MultiTaskClassifier_1"
    saved_filepath = clf.save_model(saving_path_name)

    loaded_clf = TabNetMultiTaskClassifier()
    loaded_clf.load_model(saved_filepath)

    loaded_preds = loaded_clf.predict_proba(X_test)

    loaded_test_auc = [roc_auc_score(y_score=task_pred[:, 1], y_true=y_test[:, task_idx])
                       for task_idx, task_pred in enumerate(loaded_preds[:-1])]

    print(f"FINAL AUCS SCORE FOR {dataset_name} : {loaded_test_auc}")

    assert (test_aucs == loaded_test_auc)
    assert len(test_aucs) == 4
    assert sum(test_aucs)/len(test_aucs) > 0.81

    loaded_clf.predict(X_test)

    # clf.feature_importances_

    explain_matrix, masks = clf.explain(X_test)

    fig, axs = plt.subplots(1, 3, figsize=(20, 20))

    for i in range(3):
        axs[i].imshow(masks[i][:50])
        axs[i].set_title(f"mask {i}")
