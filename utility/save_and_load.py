from utility.logger import get_logger
import os
from definitions import OUTPUT_PATH
import uuid
import glob
import pandas as pd
from classes import Window
import pickle


def save_train_test_split(train_data, test_data, dir_name):

    path_dir = os.path.join(OUTPUT_PATH, dir_name)

    if not os.path.exists(path_dir):
        get_logger().info(f'Creating dir {dir_name}.')
        os.mkdir(path_dir)

    train_path = os.path.join(path_dir, 'train')
    if not os.path.exists(train_path):
        get_logger().info(f'Train dir created.')
        os.mkdir(train_path)

    test_path = os.path.join(path_dir, 'test')
    if not os.path.exists(test_path):
        get_logger().info('Test dir created.')
        os.mkdir(test_path)

    for window in train_data:
        unique_filename = str(uuid.uuid4())
        with open(os.path.join(train_path, unique_filename), 'wb') as handle:
            pickle.dump(window, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # labels = [window.label for i in range(window.data.shape[0])]
        # window.data['label'] = labels
        # window.data.to_csv(f'{os.path.join(train_path, unique_filename)}.csv', sep=',', encoding='utf-8', index=False)

    for window in test_data:
        unique_filename = str(uuid.uuid4())
        with open(os.path.join(test_path, unique_filename), 'wb') as handle:
            pickle.dump(window, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # labels = [window.label for i in range(window.data.shape[0])]
        # window.data['label'] = labels
        # window.data.to_csv(f'{os.path.join(test_path, unique_filename)}.csv', sep=',', encoding='utf-8', index=False)

    get_logger().info(f'Saved training set of size {len(train_data)} and test set of size {len(test_data)} to disc.')


def load_train_test_split(dataset):

    path_train = os.path.join(OUTPUT_PATH, dataset, 'train/*')
    path_test = os.path.join(OUTPUT_PATH, dataset, 'test/*')

    train_names = []
    test_names = []
    for file in glob.glob(path_train, recursive=True):
        train_names.append(file)

    for file in glob.glob(path_test, recursive=True):
        test_names.append(file)

    get_logger().info(f'Found {len(train_names)} in train dir: {dataset}')
    get_logger().info(f'Found {len(test_names)} in test dir: {dataset}')

    train_windows = []
    for file in train_names:
        with open(file, 'rb') as handle:
            unserialized_data = pickle.load(handle)
            train_windows.append(unserialized_data)

    test_windows = []
    for file in test_names:
        with open(file, 'rb') as handle:
            unserialized_data = pickle.load(handle)
            test_windows.append(unserialized_data)

    get_logger().info(f'Loaded {len(train_names)+len(test_names)} windows with shape: {train_windows[0].data.shape}')
    return train_windows, test_windows


# old style of loading
def file_to_window(file):
    window = Window.Window()

    window_df = pd.read_csv(file, squeeze=True)

    window.label = window_df['label'].iloc[0]

    window_df.drop('label', axis=1, inplace=True)
    window.data = window_df

    return window
