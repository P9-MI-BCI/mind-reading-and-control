import logging
from data_preprocessing.init_dataset import format_input, print_hypothesis_options
from dispatch.dispatch_hub import dispatch
from utility.logger import get_logger
from dispatch import preliminary

"""CONFIGURATION"""
get_logger().setLevel(logging.INFO)  # Set logging level (INFO, WARNING, ERROR, CRITICAL, EXCEPTION, LOG)
# pd.set_option("display.max_rows", None, "display.max_columns", None)  # pandas print settings


def main():
    config, label_config = preliminary.check_config_files()
    preliminary.check_data_folders()
    preliminary.check_for_label_files(label_config)

    print_hypothesis_options()
    hypothesis_choice = int(input('Choose Hypothesis 1-6\n'))
    config.hypothesis_choice = hypothesis_choice

    dispatch(config)


if __name__ == '__main__':
    main()
