import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from utility import logger
import pandas as pd


def xgboost_training(X, Y):

    skf = StratifiedKFold(n_splits=5, shuffle=True)

    cv_scores = {}
    split = 0
    for train_index, val_index in skf.split(X, Y):
        model = xgb.XGBClassifier(use_label_encoder=False,
                                  objective='binary:logistic',
                                  max_depth=20,
                                  n_estimators=1000,
                                  subsample=0.5,
                                  )

        model.fit(X[train_index], Y[train_index])

        validation_prediction = model.predict(X[val_index])

        accuracy = accuracy_score(validation_prediction, Y[val_index])
        cv_scores[f'split_{split}'] = accuracy
        split += 1

    logger.get_logger().info(f'Cross Validation Complete')
    print('Cross Validation Results:')
    cv_scores = pd.DataFrame(cv_scores, index=[0])
    cv_scores['mean'] = cv_scores.mean(axis=1)
    print(f'{cv_scores}')
    logger.get_logger().info(f'Fitting entire dataset prior to online prediction')
    model = xgb.XGBClassifier(use_label_encoder=False,
                              objective='binary:logistic',
                              max_depth=20,
                              n_estimators=1000,
                              subsample=0.5,
                              )

    model.fit(X, Y)

    return model

