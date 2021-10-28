import lightgbm as lgb
from data_training.scikit_classifiers import scikit_classifier


def lightGBM_classifier_params(SEED: int):
    return {
        "objective": "binary",
        "boosting_type": "dart",
        "max_depth": 10,
        "learning_rate": 0.05,
        "bagging_seed": SEED,
        "random_state": SEED,
        "n_jobs": -1,
    }


def lgbm_classifier(train_data, test_data, channels=None, features='raw'):

    model = lgb.LGBMClassifier(**lightGBM_classifier_params(SEED=1337))

    result = scikit_classifier(model, train_data, test_data, channels, features)

    return result
