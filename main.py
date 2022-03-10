import logging
from dispatch.dispatch_hub import dispatch
from utility.logger import get_logger
from dispatch import preliminary

"""CONFIGURATION"""
get_logger().setLevel(logging.INFO)  # Set logging level (INFO, WARNING, ERROR, CRITICAL, EXCEPTION, LOG)
# pd.set_option("display.max_rows", None, "display.max_columns", None)  # pandas print settings
valid_subjects = list(range(9))


def main():
    config, label_config = preliminary.check_config_files()
    preliminary.check_data_folders()
    preliminary.check_for_label_files(label_config)

    subject = int(input('Choose subject 0-8\n'))

    if subject in valid_subjects:
        dispatch(subject, config)
    else:
        print('Input does not match subject ID')
        exit()


if __name__ == '__main__':
    main()
