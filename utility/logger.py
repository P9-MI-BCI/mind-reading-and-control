import logging
import os
import sys
from definitions import OUTPUT_PATH


'''
Logger levels:
    Debug:      10
    Info:       20
    Warning:    30
    Error:      40
    Critical:   50
'''


logger = None


def __setup_logger():
    logging_format = "%(asctime)s :: %(process)-5d :: %(levelname)-8s :: %(message)s"
    logging.basicConfig(
        format=logging_format,
        datefmt="%Y-%m-%d :: %H:%M:%S",
        level=logging.INFO,
    )

    global logger
    logger = logging.getLogger()
    return logger


def get_logger() -> logging.Logger:
    global logger
    if logger is None:
        logger = __setup_logger()

    return logger


def create_logger(name: str, level: int = logging.WARNING):

    logging_format = "%(asctime)s :: %(process)-5d :: %(levelname)s :: %(name)s :: %(message)-8s"
    date_format = "%Y-%m-%d - %H:%M:%S"
    formatter = logging.Formatter(logging_format, datefmt=date_format)

    # Create logger
    custom_logger = logging.getLogger(name)
    custom_logger.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    custom_logger.addHandler(console_handler)

    return custom_logger


class StreamToLogger(object):
    """
    Fake stream object that catches writes and writes them to the logger
    """

    def __init__(self, target_logger: logging.Logger, level: int):
        self.logger = target_logger
        self.level = level
        self.linebuff = ""

    def write(self, buff):
        for line in buff.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


def catch_output_stream_to_logger(
        target_logger: logging.Logger, level: int = logging.INFO
):
    sys.stdout = StreamToLogger(target_logger, level)
    sys.stderr = StreamToLogger(target_logger, logging.ERROR)
