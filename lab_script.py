# used for quick testing in the lab - to verify if the datasets recorded are useful

import glob
import json

import pandas as pd
import scipy.io
from classes.Dataset import Dataset
from classes.Dict import AttrDict
from scipy.signal import decimate
# hardcoded the path for lab test to be a specific folder in the project.
from data_preprocessing.data_distribution import create_uniform_distribution
from data_preprocessing.mrcp_detection import mrcp_detection_for_calibration
from data_training.scikit_classifiers import scikit_classifier_loocv_calibration, _scikit_classifier_loocv_init
from data_visualization.average_channels import average_channel, plot_average_channels


def import_dataset():
    dataset = Dataset()
    datasets = []
    for file in glob.glob('lab_dataset/*', recursive=False):
        datasets.append(scipy.io.loadmat(file))

    if len(datasets) > 1:
        exit('Too many datasets present in the lab_dataset folder.')
    else:
        dataset.data_device1 = pd.DataFrame(datasets[0]['data_device1'])
        dataset.sample_rate = 1200

    return dataset


if __name__ == "__main__":
    with open('json_configs/lab_config.json') as config_file:
        config = json.load(config_file)

    config = AttrDict(config)
    dataset = import_dataset()

    params = input('set params? y/n \n')
    if params.lower() == 'y':
        movements = int(input('enter number of movements \n'))
    else:
        movements = 30

    windows = mrcp_detection_for_calibration(data=dataset,
                                             input_peaks=movements,
                                             config=config,
                                             perfect_centering=False)

    average = average_channel(windows)
    plot_average_channels(average, config, layout='grid')

    uniform_data = create_uniform_distribution(windows)

    # make prediction
    feature = 'feature_vec'

    from sklearn import svm

    model = svm.SVC()

    data, features, labels, channels = _scikit_classifier_loocv_init(uniform_data,
                                                                     config.EEG_Channels,
                                                                     prediction='w',
                                                                     features=feature
                                                                     )
    res = scikit_classifier_loocv_calibration(model,
                                              data,
                                              feats=features,
                                              labels=labels,
                                              channels=[config.EEG_Channels]
                                              )

    print(res[-1])
