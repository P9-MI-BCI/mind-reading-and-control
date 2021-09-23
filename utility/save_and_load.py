from utility.logger import get_logger
import os
from definitions import OUTPUT_PATH
import uuid

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
