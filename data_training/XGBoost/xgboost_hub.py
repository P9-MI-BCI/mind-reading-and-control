import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def xgboost_training(X, Y, online_X, online_Y):

    skf = StratifiedKFold(n_splits=5, shuffle=True)

    for train_index, val_index in skf.split(X, Y):
        model = xgb.XGBClassifier(use_label_encoder=False, objective='binary:logistic', max_depth=20, n_estimators=1000)

        model.fit(X[train_index], Y[train_index])

        validation_prediction = model.predict(X[val_index])

        accuracy = accuracy_score(validation_prediction, Y[val_index])
        print(accuracy)

    print('online test')
    model = xgb.XGBClassifier(use_label_encoder=False, objective='binary:logistic', max_depth=20, n_estimators=1000)

    model.fit(X, Y)

    online_prediction = model.predict(online_X)

    accuracy = accuracy_score(online_prediction, online_Y)

    print(accuracy)
