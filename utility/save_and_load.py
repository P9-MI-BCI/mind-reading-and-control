import os
import shutil
import uuid
import glob
import pickle
import pandas as pd
from classes import Window
from definitions import OUTPUT_PATH
from utility.logger import get_logger


def save_train_test_split(train_data, test_data, dir_name: str):
    path_dir = os.path.join(OUTPUT_PATH, dir_name)
    unique_filename = str(uuid.uuid4())

    old_windows_path = os.path.join(OUTPUT_PATH, 'old_windows')
    if not os.path.exists(old_windows_path):
        os.mkdir(old_windows_path)

    # Creating parent EEG dir + train and test subdirs at output path. Overwrites if already exist
    if not os.path.exists(path_dir):
        get_logger().info(f'Creating new dir {dir_name} at {path_dir}.')
        os.mkdir(path_dir)
    else:
        get_logger().info(
            f'File dir {dir_name} already exists, renaming old directory to {os.path.join(old_windows_path, unique_filename)}')
        os.rename(path_dir, os.path.join(old_windows_path, unique_filename))
        os.mkdir(path_dir)

    train_path = os.path.join(path_dir, 'train')
    if os.path.exists(train_path):
        get_logger().info(f'Train dir overwritten at {path_dir}')
        shutil.rmtree(path_dir)
    else:
        get_logger().info(f'Train dir created at {train_path}')
        os.mkdir(train_path)

    test_path = os.path.join(path_dir, 'test')
    if os.path.exists(test_path):
        get_logger().info(f'Train dir overwritten at {path_dir}')
        shutil.rmtree(path_dir)
    else:
        get_logger().info(f'Test dir created at {test_path}')
        os.mkdir(test_path)

    # Writing train and test split files to train and test dir.
    for window in train_data:
        unique_filename = str(uuid.uuid4())
        with open(os.path.join(train_path, unique_filename), 'wb') as handle:
            pickle.dump(window, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for window in test_data:
        unique_filename = str(uuid.uuid4())
        with open(os.path.join(test_path, unique_filename), 'wb') as handle:
            pickle.dump(window, handle, protocol=pickle.HIGHEST_PROTOCOL)

    get_logger().info(f'Saved training set of size {len(train_data)} and test set of size {len(test_data)} to disc.')


def load_train_test_split(dir_name: str):
    path_train = os.path.join(OUTPUT_PATH, dir_name, 'train/*')
    path_test = os.path.join(OUTPUT_PATH, dir_name, 'test/*')

    train_names = []
    test_names = []
    for file in glob.glob(path_train, recursive=True):
        train_names.append(file)

    for file in glob.glob(path_test, recursive=True):
        test_names.append(file)

    get_logger().info(f'Found {len(train_names)} in train dir: {dir_name}')
    get_logger().info(f'Found {len(test_names)} in test dir: {dir_name}')

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

    get_logger().info(
        f'Loaded {len(train_names) + len(test_names)} windows with shape: {train_windows[0].data.shape}')
    return train_windows, test_windows
