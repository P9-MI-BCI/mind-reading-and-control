import pandas as pd
import json
import logging
from classes.Dict import AttrDict
from data_preprocessing.init_dataset import get_datasets
from dispatch.dispatch_hub import dispatch
from utility.logger import get_logger

"""CONFIGURATION"""
get_logger().setLevel(logging.INFO)  # Set logging level (INFO, WARNING, ERROR, CRITICAL, EXCEPTION, LOG)
# pd.set_option("display.max_rows", None, "display.max_columns", None)  # pandas print settings


def main():
    # with open('json_configs/default.json') as c_config:
    #     config = AttrDict(json.load(c_config))
    #
    # config = AttrDict(config)
    #
    # valid_subjects = list(range(9))
    # subject = int(input('Choose subject 0-8\n'))
    #
    # if subject in valid_subjects:
    #     dwell_dataset, online_dataset, training_dataset = get_datasets(subject_id=subject, config=config)
    # else:
    #     print('Input does not match subject ID')
    #     exit()

    with open('config.json') as config_file, open('script_parameters.json') as script_parameters:
        config = json.load(config_file)['cue_set1']  # Choose config
        script_params = json.load(script_parameters)  # Load script parameters

    script_params = AttrDict(script_params)
    config = AttrDict(config)

    dispatch(script_params, config)


if __name__ == '__main__':
    main()
