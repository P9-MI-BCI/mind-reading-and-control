from data_preprocessing.emg_processing import onset_detection
from data_preprocessing.init_dataset import init
import glob
import pandas as pd
import scipy.io
import os
from classes.Dataset import Dataset
from definitions import DATASET_PATH
from utility.logger import get_logger


def create_dataset_from_config(config, label_config):
    data = []
    names = []
    filenames = []
    for k in label_config:
        path = os.path.join(DATASET_PATH, k, '**/*.mat')
        for file in glob.glob(path, recursive=True):
            for trial in label_config[k]['emg_outliers']:
                if file.lower().endswith(trial.lower()):
                    data.append(scipy.io.loadmat(file))
                    filenames.append(file.lower().split('\\')[-3] + ' ' + file.lower().split('\\')[-1])
                    if 'close' in file:
                        names.append(0)
                    elif 'open' in file:
                        names.append(1)
                    else:
                        names.append(None)

    if len(data) == 0:
        get_logger().error(f'No files found in {path}')
    else:
        train_data = []
        assert len(data) == len(names)
        for dataset, label, filename in zip(data, names, filenames):
            train_data.append(init(dataset, config, label, filename))
        return train_data


def outlier_test(config, label_config, gridsearch=False):
    data = create_dataset_from_config(config, label_config)
    grid_results = {}
    if (gridsearch):
        for static_clusters in (True, False):
            for proximity_outliers in (True, False):
                    for normalization in (True, False):
                        print(f'Trying Static clustering: {static_clusters} '
                              f'Proximity outlier removal: {proximity_outliers} '
                              f'Normalization: {normalization}')
                        for dataset in data:
                            try:
                                onset_detection(dataset, config, static_clusters=static_clusters,
                                                proximity_outliers=proximity_outliers, normalization=normalization)
                                assert len(dataset.clusters) == 20
                            except AssertionError:
                                get_logger().warning(
                                    f"{dataset.filename} contains {len(dataset.clusters)} clusters (not 20) "
                                    f"with current outlier parameters")
                                # get_logger().debug([f'Static clustering: {static_clusters}',
                                #                     f'Proximity outlier removal: {proximity_outliers}',
                                #                     f'Normalization: {normalization}',
                                #                     f'Filename: {dataset.filename}'])
                                if ((static_clusters, proximity_outliers, normalization) in grid_results):
                                    grid_results[(static_clusters, proximity_outliers, normalization)] += 1
                                else:
                                    grid_results[(static_clusters, proximity_outliers, normalization)] = 1
        print('Results of outlier gridsearch:\n')
        for k, v in grid_results.items():
            print(f'Static clustering: {k[0]}, Proximity outlier merging: {k[1]}, Normalization: {k[2]}, '
                  f'Datasets where clusters != 20: {v}')

    # try:
    else:
        bad_datasets = []
        missed_clusters = 0
        for dataset in data:
            try:
                onset_detection(dataset, config,
                                static_clusters=True,
                                proximity_outliers=True,
                                normalization=True)
                assert len(dataset.clusters) == 20
            except AssertionError:
                get_logger().debug(f"{dataset.filename} contains {len(dataset.clusters)} clusters (not 20) "
                                   f"with current outlier parameters")
                bad_datasets.append(dataset)
        get_logger().debug('-' * 65)
        get_logger().debug(f'Outier testing completed. Total bad datasets: {len(bad_datasets)}. Bad datasets were:')
        for dataset in bad_datasets:
            missed_clusters += 20 - len(dataset.clusters)
            get_logger().debug(f'{dataset.filename} with {len(dataset.clusters)} clusters')
        get_logger().debug(f'Total amount of potentially missed clusters: {missed_clusters}')
    # except TypeError:
    #    get_logger().error("Dataset for EMG outlier detection is probably Nonetype, fix path/config file")
