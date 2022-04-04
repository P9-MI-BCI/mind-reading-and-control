import logging
from data_preprocessing.init_dataset import format_input
from data_preprocessing.outlier_testing import outlier_test
from dispatch.dispatch_hub import dispatch
from utility.logger import get_logger
from dispatch import preliminary

"""CONFIGURATION"""
get_logger().setLevel(logging.DEBUG)  # Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL, EXCEPTION, LOG)
logging.getLogger('matplotlib.font_manager').disabled = True
# pd.set_option("display.max_rows", None, "display.max_columns", None)  # pandas print settings
valid_subjects = list(range(9))


def main():
    config, label_config = preliminary.check_config_files()
    preliminary.check_data_folders()
    preliminary.check_for_label_files(label_config)

    outlier_test_suite = True  # input('Run outlier test suite (y/n)\n')

    subject = int(input('Choose subject 0-8\n'))
    #
    # include_all_subjects = input('Include all subjects for classification (y/n)\n')
    # config.include_all_subjects = format_input(include_all_subjects)
    #
    # include_rest = input('Binary rest/movement classification (y/n)\n')
    # config.rest_classification = format_input(include_rest)

    # if (True):#format_input(outlier_test_suite)):
    #     get_logger().setLevel(logging.DEBUG)
    #     outlier_test(config, label_config)

    if subject in valid_subjects:
        dispatch(subject, config)
    else:
        print('Input does not match subject ID')
        exit()


if __name__ == '__main__':
    main()
