def test_imports():
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor  # noqa
    from pytorch_tabnet.multitask import TabNetMultiTaskClassifier  # noqa
    from pytorch_tabnet.pretraining import TabNetPretrainer  # noqa

    assert False


if __name__ == "__main__":
    test_imports()
