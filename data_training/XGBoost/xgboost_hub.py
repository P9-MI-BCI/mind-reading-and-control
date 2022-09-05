import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from utility import logger
import pandas as pd
from matplotlib import pyplot
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from utility.logger import result_logger

space = {'max_depth': hp.quniform("max_depth", 3, 18, 1),
         'gamma': hp.uniform('gamma', 1, 9),
         'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
         'reg_lambda': hp.uniform('reg_lambda', 0, 1),
         'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
         'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
         'subsample': hp.uniform('subsample', 0.5, 1),
         'n_estimators': 1000,
         'seed': 0
         }

params= {}


def xgboost_training(X, Y, logger_location=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    cv_scores = {}
    split = 0
    epochs_avg = []
    for train_index, val_index in skf.split(X, Y):
        model = xgb.XGBClassifier(use_label_encoder=False,
                                  objective='binary:logistic',
                                  max_depth=13,
                                  n_estimators=10000,
                                  subsample=0.5,
                                  booster='gbtree',
                                  learning_rate=0.015,
                                  gamma=1,
                                  colsample_bytree=0.9
                                  )

        eval_set = [(X[train_index], Y[train_index]), (X[val_index], Y[val_index])]
        model = model.fit(X[train_index], Y[train_index], eval_metric=["error", "auc"], eval_set=eval_set,
                          verbose=False, early_stopping_rounds=10)

        validation_prediction = model.predict(X[val_index])
        results = model.evals_result()
        epochs = len(results['validation_0']['error'])

        epochs_avg.append(epochs)

        if logger.get_logger().level == 10:
            x_axis = range(0, epochs)
            fig, ax = pyplot.subplots(figsize=(12, 12))
            ax.plot(x_axis, results['validation_0']['auc'], label='Train')
            ax.plot(x_axis, results['validation_1']['auc'], label='Test')
            ax.legend()

            pyplot.ylabel('AUC')
            pyplot.show()
        accuracy = accuracy_score(validation_prediction, Y[val_index])
        cv_scores[f'split_{split}'] = accuracy
        split += 1

    logger.get_logger().info(f'Cross Validation Complete')
    print('Cross Validation Results:')
    cv_scores = pd.DataFrame(cv_scores, index=[0])
    cv_scores['mean'] = cv_scores.mean(axis=1)
    cv_scores['std'] = cv_scores.std(axis=1)
    print(f'{cv_scores}')
    logger.get_logger().info(f'Fitting entire dataset prior to online prediction')
    model = xgb.XGBClassifier(use_label_encoder=False,
                              objective='binary:logistic',
                              max_depth=13,
                              n_estimators=int(sum(epochs_avg)/len(epochs_avg)),
                              subsample=1,
                              booster='gbtree',
                              learning_rate=0.015,
                              gamma=1,
                              colsample_bytree=0.9
                              )

    model.fit(X, Y, verbose=False)

    if logger_location is not None:
        result_logger(logger_location, f'Training 5 fold cross validation.\n')
        result_logger(logger_location, f'{cv_scores}\n')

    return model


def optimized_xgboost(X, Y):
    def training_objective(params):
        skf = StratifiedKFold(n_splits=5, shuffle=True)

        cv_scores = {}
        split = 0

        for train_index, val_index in skf.split(X, Y):
            model = xgb.XGBClassifier(use_label_encoder=False,
                                      objective='binary:logistic',
                                      n_estimators=params['n_estimators'],
                                      max_depth=int(params['max_depth']),
                                      # gamma=params['gamma'],
                                      # reg_alpha=int(params['reg_alpha']),
                                      # min_child_weight=int(params['min_child_weight']),
                                      # colsample_bytree=int(params['colsample_bytree']),
                                      booster='gbtree',
                                      subsample=params['subsample'],
                                      learning_rate=0.01)

            eval_set = [(X[train_index], Y[train_index]), (X[val_index], Y[val_index])]
            model = model.fit(X[train_index], Y[train_index],
                              eval_metric=["auc"],
                              eval_set=eval_set,
                              early_stopping_rounds=10,
                              verbose=False)


            validation_prediction = model.predict(X[val_index])
            accuracy = accuracy_score(validation_prediction, Y[val_index])
            cv_scores[f'split_{split}'] = accuracy

        cv_scores = pd.DataFrame(cv_scores, index=[0])
        cv_scores['mean'] = cv_scores.mean(axis=1)
        print(cv_scores)
        return {'loss': -cv_scores['mean'], 'status': STATUS_OK}

    trials = Trials()

    best_hyperparams = fmin(fn=training_objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=50,
                            trials=trials)

    print(best_hyperparams)

