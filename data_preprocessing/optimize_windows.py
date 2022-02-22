from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from data_training.scikit_classifiers import scikit_classifier_loocv_calibration, _scikit_classifier_loocv_init
from data_training.util import multiproc_classifier
from utility.logger import get_logger


def optimize_channels(data, model, channels):
    feature = 'feature_vec'

    if model == 'knn':
        model = KNeighborsClassifier(n_neighbors=3)
    elif model == 'svm':
        model = svm.SVC()
    elif model == 'lda':
        model = LinearDiscriminantAnalysis()
    else:
        get_logger().info(f'Model is not supported {model}')

    preds_data, feats, labels, channels = _scikit_classifier_loocv_init(data=data,
                                                                        channels=channels,
                                                                        features=feature,
                                                                        prediction='w')

    results = multiproc_classifier(model, scikit_classifier_loocv_calibration, channels, preds_data, feats, labels)

    best_score = 0
    best_score_dict = 0
    best_model = 0
    best_channel_combination = 0

    for result in results:
        if result[0] > best_score:
            best_score = result[0]
            best_model = result[1]
            best_channel_combination = result[2]
            best_score_dict = result[3]

    return best_score_dict, best_model, best_channel_combination
