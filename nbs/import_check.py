def test_imports():
    from pytorch_tabnet.tab_models.tab_model import TabNetClassifier, TabNetRegressor  # noqa
    from pytorch_tabnet.tab_models.multitask import TabNetMultiTaskClassifier  # noqa
    from pytorch_tabnet.tab_models.pretraining import TabNetPretrainer  # noqa

    assert True


if __name__ == "__main__":
    test_imports()
