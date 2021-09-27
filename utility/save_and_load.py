from utility.logger import get_logger
import os
from definitions import OUTPUT_PATH
import uuid
import glob
import pandas as pd
from classes import Frame


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

    for frame in train_data:
        unique_filename = str(uuid.uuid4())
        labels = [frame.label for i in range(frame.data.shape[0])]
        frame.data['label'] = labels
        frame.data.to_csv(f'{os.path.join(train_path, unique_filename)}.csv', sep=',', encoding='utf-8', index=False)

    for frame in test_data:
        unique_filename = str(uuid.uuid4())
        labels = [frame.label for i in range(frame.data.shape[0])]
        frame.data['label'] = labels
        frame.data.to_csv(f'{os.path.join(test_path, unique_filename)}.csv', sep=',', encoding='utf-8', index=False)

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

    train_frames = []
    for file in train_names:
        train_frames.append(file_to_frame(file))

    test_frames = []
    for file in test_names:
        test_frames.append(file_to_frame(file))

    get_logger().info(f'Loaded {len(train_names)+len(test_names)} frames with shape: {train_frames[0].data.shape}')
    return train_frames, test_frames


def file_to_frame(file):
    frame = Frame.Frame()

    frame_df = pd.read_csv(file, squeeze=True)

    frame.label = frame_df['label'].iloc[0]

    frame_df.drop('label', axis=1, inplace=True)
    frame.data = frame_df

    return frame
