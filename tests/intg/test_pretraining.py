



def test_pretraining():

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

    X_train = train[features].values[train_indices]
    y_train = train[target].values[train_indices]

    X_valid = train[features].values[valid_indices]
    y_valid = train[target].values[valid_indices]

    X_test = train[features].values[test_indices]
    y_test = train[target].values[test_indices]

    from pytorch_tabnet.pretraining import TabNetPretrainer

    unsupervised_model = TabNetPretrainer(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type='entmax',  # "sparsemax",
        n_shared_decoder=1,  # nb shared glu for decoding
        n_indep_decoder=1,  # nb independent glu for decoding
        #     grouped_features=[[0, 1]], # you can group features together here
        # verbose=5,
    )


    max_epochs =  20  # 1000
    unsupervised_model.fit(
        X_train=X_train,
        eval_set=[X_valid],
        max_epochs=max_epochs, patience=5,
        batch_size=2048, #virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        pretraining_ratio=0.5,
    )

    score_unsup = unsupervised_model.best_cost
    assert score_unsup<2.1
    reconstructed_X, embedded_X = unsupervised_model.predict(X_valid)
    assert (reconstructed_X.shape == embedded_X.shape)
    unsupervised_explain_matrix, unsupervised_masks = unsupervised_model.explain(X_valid)
    fig, axs = plt.subplots(1, 3, figsize=(20, 20))

    for i in range(3):
        axs[i].imshow(unsupervised_masks[i][:50])
        axs[i].set_title(f"mask {i}")


    unsupervised_model.save_model('./test_pretrain')
    loaded_pretrain = TabNetPretrainer()
    loaded_pretrain.load_model('./test_pretrain.zip')

    clf = TabNetClassifier(optimizer_fn=torch.optim.Adam,
                           optimizer_params=dict(lr=2e-3),
                           scheduler_params={"step_size": 10,  # how to use learning rate scheduler
                                             "gamma": 0.9},
                           scheduler_fn=torch.optim.lr_scheduler.StepLR,
                           mask_type='sparsemax',  # This will be overwritten if using pretrain model
                           # verbose=5,
                           )
    max_epochs =  5  # 1000

    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'],
        eval_metric=['auc'],
        max_epochs=max_epochs, patience=20,
        batch_size=1024,# virtual_batch_size=128,
        num_workers=0,
        weights=1,
        drop_last=False,
        from_unsupervised=loaded_pretrain,

    )

    plt.plot(clf.history['loss'])

    plt.plot(clf.history['train_auc'])
    plt.plot(clf.history['valid_auc'])

    plt.plot(clf.history['lr'])

    preds = clf.predict_proba(X_test)
    test_auc = roc_auc_score(y_score=preds[:, 1], y_true=y_test)

    preds_valid = clf.predict_proba(X_valid)
    valid_auc = roc_auc_score(y_score=preds_valid[:, 1], y_true=y_valid)

    print(f"BEST VALID SCORE FOR {dataset_name} : {clf.best_cost}")
    assert clf.best_cost>0.7

    print(f"FINAL TEST SCORE FOR {dataset_name} : {test_auc}")
    assert test_auc>0.7

    assert np.isclose(valid_auc, np.max(clf.history['valid_auc']), atol=1e-2)

    saving_path_name = "./tabnet_model_test_1"
    saved_filepath = clf.save_model(saving_path_name)

    loaded_clf = TabNetClassifier()
    loaded_clf.load_model(saved_filepath)

    loaded_preds = loaded_clf.predict_proba(X_test)
    loaded_test_auc = roc_auc_score(y_score=loaded_preds[:, 1], y_true=y_test)

    print(f"FINAL TEST SCORE FOR {dataset_name} : {loaded_test_auc}")

    assert (test_auc == loaded_test_auc)

    # clf.feature_importances_

    explain_matrix, masks = clf.explain(X_test)

    fig, axs = plt.subplots(1, 3, figsize=(20, 20))

    for i in range(3):
        axs[i].imshow(masks[i][:50])
        axs[i].set_title(f"mask {i}")
