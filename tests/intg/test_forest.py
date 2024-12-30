def test_forest():
    from pytorch_tabnet.tab_model import TabNetClassifier

    import torch
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    np.random.seed(0)

    import os
    import wget
    from pathlib import Path
    import shutil
    import gzip

    from matplotlib import pyplot as plt

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    dataset_name = 'forest-cover-type'
    tmp_out = Path('./data/' + dataset_name + '.gz')
    out = Path(os.getcwd() + '/data/' + dataset_name + '.csv')

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print("File already exists.")
    else:
        print("Downloading file...")
        wget.download(url, tmp_out.as_posix())
        with gzip.open(tmp_out, 'rb') as f_in:
            with open(out, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    target = "Covertype"

    bool_columns = [
        "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3",
        "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4",
        "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9",
        "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14",
        "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19",
        "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24",
        "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29",
        "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34",
        "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39",
        "Soil_Type40"
    ]

    int_columns = [
        "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
        "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points"
    ]

    feature_columns = (
            int_columns + bool_columns + [target])

    train = pd.read_csv(out, header=None, names=feature_columns)

    n_total = len(train)

    train_val_indices, test_indices = train_test_split(
        range(n_total), test_size=0.2, random_state=0)
    train_indices, valid_indices = train_test_split(
        train_val_indices, test_size=0.2 / 0.6, random_state=0)

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

    unused_feat = []

    features = [col for col in train.columns if col not in unused_feat + [target]]

    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]

    cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    clf = TabNetClassifier(
        n_d=64, n_a=64, n_steps=5,
        gamma=1.5, n_independent=2, n_shared=2,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=1,
        lambda_sparse=1e-4, momentum=0.3, clip_value=2.,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"gamma": 0.95,
                          "step_size": 20},
        scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15
    )

    if os.getenv("CI", False):
        X_train = train[features].values[train_indices][:1000, :]
        y_train = train[target].values[train_indices][:1000]
    else:
        X_train = train[features].values[train_indices]
        y_train = train[target].values[train_indices]

    X_valid = train[features].values[valid_indices]
    y_valid = train[target].values[valid_indices]

    X_test = train[features].values[test_indices]
    y_test = train[target].values[test_indices]

    max_epochs = 5

    from pytorch_tabnet.augmentations import ClassificationSMOTE
    aug = ClassificationSMOTE(p=0.2)

    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'],
        max_epochs=max_epochs, patience=100,
        batch_size=16384, virtual_batch_size=256,
        augmentations=aug
    )

    plt.plot(clf.history['loss'])

    plt.plot(clf.history['train_accuracy'])
    plt.plot(clf.history['valid_accuracy'])

    preds_mapper = {idx: class_name for idx, class_name in enumerate(clf.classes_)}

    preds = clf.predict_proba(X_test)

    y_pred = np.vectorize(preds_mapper.get)(np.argmax(preds, axis=1))

    test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)

    print(f"BEST VALID SCORE FOR {dataset_name} : {clf.best_cost}")
    print(f"FINAL TEST SCORE FOR {dataset_name} : {test_acc}")

    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    print(f"FINAL TEST SCORE FOR {dataset_name} : {test_acc}")

    saved_filename = clf.save_model('test_model')

    loaded_clf = TabNetClassifier()
    loaded_clf.load_model(saved_filename)

    loaded_preds = loaded_clf.predict_proba(X_test)
    loaded_y_pred = np.vectorize(preds_mapper.get)(np.argmax(loaded_preds, axis=1))

    loaded_test_acc = accuracy_score(y_pred=loaded_y_pred, y_true=y_test)

    print(f"FINAL TEST SCORE FOR {dataset_name} : {loaded_test_acc}")

    assert (test_acc == loaded_test_acc)
    assert test_acc > 0.2

    clf.feature_importances_

    explain_matrix, masks = clf.explain(X_test)

    fig, axs = plt.subplots(1, 5, figsize=(20, 20))

    for i in range(5):
        axs[i].imshow(masks[i][:50])
        axs[i].set_title(f"mask {i}")
