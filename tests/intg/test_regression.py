






def test_regression():
    from pytorch_tabnet.tab_model import TabNetRegressor

    import torch
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_squared_error

    import pandas as pd
    import numpy as np
    np.random.seed(0)

    import os
    import wget
    from pathlib import Path

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
    categorical_columns = []
    categorical_dims = {}
    for col in train.columns[train.dtypes == object]:
        print(col, train[col].nunique())
        l_enc = LabelEncoder()
        train[col] = train[col].fillna("VV_likely")
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)

    for col in train.columns[train.dtypes == 'float64']:
        train.fillna(train.loc[train_indices, col].mean(), inplace=True)
    # %% md
    unused_feat = ['Set']

    features = [col for col in train.columns if col not in unused_feat + [target]]

    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]

    cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    cat_emb_dim = [5, 4, 3, 6, 2, 2, 1, 10]
    clf = TabNetRegressor(cat_dims=cat_dims, cat_emb_dim=cat_emb_dim, cat_idxs=cat_idxs)
    X_train = train[features].values[train_indices]
    y_train = train[target].values[train_indices].reshape(-1, 1)

    X_valid = train[features].values[valid_indices]
    y_valid = train[target].values[valid_indices].reshape(-1, 1)

    X_test = train[features].values[test_indices]
    y_test = train[target].values[test_indices].reshape(-1, 1)
    # %%
    max_epochs =  5
    # %%
    from pytorch_tabnet.augmentations import RegressionSMOTE
    aug = RegressionSMOTE(p=0.2)
    # %%
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'],
        eval_metric=['rmsle', 'mae', 'rmse', 'mse'],
        max_epochs=max_epochs,
        patience=50,

        batch_size=1024, #virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        augmentations=aug,  # aug
    )
    # %%
    preds = clf.predict(X_test)

    y_true = y_test

    test_score = mean_squared_error(y_pred=preds, y_true=y_true)

    print(f"BEST VALID SCORE FOR {dataset_name} : {clf.best_cost}")
    print(f"FINAL TEST SCORE FOR {dataset_name} : {test_score}")
    # %% md
    saving_path_name = "./tabnet_model_test_1"
    saved_filepath = clf.save_model(saving_path_name)
    loaded_clf = TabNetRegressor()
    loaded_clf.load_model(saved_filepath)
    loaded_preds = loaded_clf.predict(X_test)
    loaded_test_mse = mean_squared_error(loaded_preds, y_test)

    print(f"FINAL TEST SCORE FOR {dataset_name} : {loaded_test_mse}")
    # %%
    assert (test_score == loaded_test_mse)
    assert test_score<0.29
    # clf.feature_importances_
    explain_matrix, masks = clf.explain(X_test)
    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(20, 20))

    for i in range(3):
        axs[i].imshow(masks[i][:50])
        axs[i].set_title(f"mask {i}")
