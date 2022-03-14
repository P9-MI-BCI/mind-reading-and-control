import json
import logging
from classes.Dict import AttrDict
from data_preprocessing.init_dataset import format_input
from dispatch.dispatch_hub import dispatch
from utility.logger import get_logger
from dispatch import preliminary
import os

"""CONFIGURATION"""
get_logger().setLevel(logging.INFO)  # Set logging level (INFO, WARNING, ERROR, CRITICAL, EXCEPTION, LOG)
# pd.set_option("display.max_rows", None, "display.max_columns", None)  # pandas print settings
valid_subjects = list(range(9))


def main():
    preliminary.check_data_folders()

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'json_configs/default.json')

    with open(filename) as c_config:
        config = AttrDict(json.load(c_config))

    subject = int(input('Choose subject 0-8\n'))

    transfer_learning = input('Enable Transfer Learning (y/n)\n')
    config.transfer_learning = format_input(transfer_learning)

    include_rest = input('Binary rest/movement classification (y/n)\n')
    config.rest_classification = format_input(include_rest)

    if subject in valid_subjects:
        dispatch(subject, config)
    else:
        print('Input does not match subject ID')
        exit()


if __name__ == '__main__':
    main()
