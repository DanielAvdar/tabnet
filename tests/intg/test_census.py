




def test_census():
    # %%
    from pytorch_tabnet.tab_model import TabNetClassifier

    import torch
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import roc_auc_score

    import pandas as pd
    import numpy as np
    np.random.seed(0)

    import scipy

    import os
    import wget
    from pathlib import Path

    from matplotlib import pyplot as plt
    # %%
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = f"1"
    # %%
    import torch
    torch.__version__
    torch.cuda.is_available()
    # %% md
    # # Download census-income dataset
    # %%
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    dataset_name = 'census-income'
    out = Path(os.getcwd() + '/data/' + dataset_name + '.csv')
    # %%
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print("File already exists.")
    else:
        print("Downloading file...")
        wget.download(url, out.as_posix())
    # %% md
    # # Load data and split
    # %%
    train = pd.read_csv(out)
    target = ' <=50K'
    if "Set" not in train.columns:
        train["Set"] = np.random.choice(["train", "valid", "test"], p=[.8, .1, .1], size=(train.shape[0],))

    train_indices = train[train.Set == "train"].index
    valid_indices = train[train.Set == "valid"].index
    test_indices = train[train.Set == "test"].index
    # %% md
    # # Simple preprocessing
    #
    # Label encode categorical features and fill empty cells.
    # %%
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
    # %%
    # check that pipeline accepts strings
    train.loc[train[target] == 0, target] = "wealthy"
    train.loc[train[target] == 1, target] = "not_wealthy"
    # %% md
    # # Define categorical features for categorical embeddings
    # %%
    unused_feat = ['Set']

    features = [col for col in train.columns if col not in unused_feat + [target]]

    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]

    cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    # %% md
    # # Grouped features
    #
    # You can now specify groups of feature which will share a common attention.
    #
    # This may be very usefull for features comming from a same preprocessing technique like PCA for example.
    # %%
    len(features)
    # %%
    grouped_features = [[0, 1, 2], [8, 9, 10]]
    # %% md
    # # Network parameters
    # %%
    tabnet_params = {"cat_idxs": cat_idxs,
                     "cat_dims": cat_dims,
                     "cat_emb_dim": 2,
                     "optimizer_fn": torch.optim.Adam,
                     "optimizer_params": dict(lr=2e-2),
                     "scheduler_params": {"step_size": 50,  # how to use learning rate scheduler
                                          "gamma": 0.9},
                     "scheduler_fn": torch.optim.lr_scheduler.StepLR,
                     "mask_type": 'entmax',  # "sparsemax"
                     "grouped_features": grouped_features
                     }

    clf = TabNetClassifier(**tabnet_params
                           )
    clf.device
    # %% md
    # # Training
    # %%
    X_train = train[features].values[train_indices]
    y_train = train[target].values[train_indices]

    X_valid = train[features].values[valid_indices]
    y_valid = train[target].values[valid_indices]

    X_test = train[features].values[test_indices]
    y_test = train[target].values[test_indices]
    # %%
    max_epochs = 50 if not os.getenv("CI", False) else 2
    # %%
    from pytorch_tabnet.augmentations import ClassificationSMOTE
    aug = ClassificationSMOTE(p=0.2)
    # %%
    # This illustrates the behaviour of the model's fit method using Compressed Sparse Row matrices
    sparse_X_train = scipy.sparse.csr_matrix(X_train)  # Create a CSR matrix from X_train
    sparse_X_valid = scipy.sparse.csr_matrix(X_valid)  # Create a CSR matrix from X_valid

    # Fitting the model
    clf.fit(
        X_train=sparse_X_train, y_train=y_train,
        eval_set=[(sparse_X_train, y_train), (sparse_X_valid, y_valid)],
        eval_name=['train', 'valid'],
        eval_metric=['auc'],
        max_epochs=max_epochs, patience=20,
        batch_size=1024, virtual_batch_size=128,
        num_workers=0,
        weights=1,
        drop_last=False,
        augmentations=aug,  # aug, None
    )
    # %%
    # This illustrates the warm_start=False behaviour
    save_history = []

    # Fitting the model without starting from a warm start nor computing the feature importance
    for _ in range(2):
        clf.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_name=['train', 'valid'],
            eval_metric=['auc'],
            max_epochs=max_epochs, patience=20,
            batch_size=1024, virtual_batch_size=128,
            num_workers=0,
            weights=1,
            drop_last=False,
            augmentations=aug,  # aug, None
            compute_importance=False
        )
        save_history.append(clf.history["valid_auc"])

    assert (np.all(np.array(save_history[0] == np.array(save_history[1]))))

    save_history = []  # Resetting the list to show that it also works when computing feature importance

    # Fitting the model without starting from a warm start but with the computing of the feature importance activated
    for _ in range(2):
        clf.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_name=['train', 'valid'],
            eval_metric=['auc'],
            max_epochs=max_epochs, patience=20,
            batch_size=1024, virtual_batch_size=128,
            num_workers=0,
            weights=1,
            drop_last=False,
            augmentations=aug,  # aug, None
            compute_importance=True  # True by default so not needed
        )
        save_history.append(clf.history["valid_auc"])

    assert (np.all(np.array(save_history[0] == np.array(save_history[1]))))
    # %%
    # plot losses
    plt.plot(clf.history['loss'])
    # %%
    # plot auc
    plt.plot(clf.history['train_auc'])
    plt.plot(clf.history['valid_auc'])
    # %%
    # plot learning rates
    plt.plot(clf.history['lr'])
    # %% md
    # ## Predictions
    # %%
    preds = clf.predict_proba(X_test)
    test_auc = roc_auc_score(y_score=preds[:, 1], y_true=y_test)

    preds_valid = clf.predict_proba(X_valid)
    valid_auc = roc_auc_score(y_score=preds_valid[:, 1], y_true=y_valid)

    print(f"BEST VALID SCORE FOR {dataset_name} : {clf.best_cost}")
    print(f"FINAL TEST SCORE FOR {dataset_name} : {test_auc}")
    # %%
    # check that best weights are used
    assert np.isclose(valid_auc, np.max(clf.history['valid_auc']), atol=1e-6)
    # %%
    clf.predict(X_test)
    # %% md
    # # Save and load Model
    # %%
    # save tabnet model
    saving_path_name = "./tabnet_model_test_1"
    saved_filepath = clf.save_model(saving_path_name)
    # %%
    # define new model with basic parameters and load state dict weights
    loaded_clf = TabNetClassifier()
    loaded_clf.load_model(saved_filepath)
    # %%
    loaded_preds = loaded_clf.predict_proba(X_test)
    loaded_test_auc = roc_auc_score(y_score=loaded_preds[:, 1], y_true=y_test)

    print(f"FINAL TEST SCORE FOR {dataset_name} : {loaded_test_auc}")
    # %%
    assert (test_auc == loaded_test_auc)
    # %%
    loaded_clf.predict(X_test)
    # %% md
    # # Global explainability : feat importance summing to 1
    # %%
    clf.feature_importances_
    # %% md
    # # Local explainability and masks
    # %%
    explain_matrix, masks = clf.explain(X_test)
    # %%
    fig, axs = plt.subplots(1, 3, figsize=(20, 20))

    for i in range(3):
        axs[i].imshow(masks[i][:50])
        axs[i].set_title(f"mask {i}")
        axs[i].set_xticklabels(labels=features, rotation=45)
    assert False