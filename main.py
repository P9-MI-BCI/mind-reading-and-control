import pandas as pd
import json
import logging
from classes.Dict import AttrDict
from dispatch.dispatch_hub import dispatch
from utility.logger import get_logger

"""CONFIGURATION"""
get_logger().setLevel(logging.INFO)  # Set logging level (INFO, WARNING, ERROR, CRITICAL, EXCEPTION, LOG)
# pd.set_option("display.max_rows", None, "display.max_columns", None)  # pandas print settings


def main():
    with open('config.json') as config_file, open('script_parameters.json') as script_parameters:
        config = json.load(config_file)['cue_set0']  # Choose config
        script_params = json.load(script_parameters)  # Load script parameters

    script_params = AttrDict(script_params)
    config = AttrDict(config)

    dispatch(script_params, config)


if __name__ == '__main__':
    main()
