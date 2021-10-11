import logging
import os
import sys
from definitions import OUTPUT_PATH

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


def create_logger(name: str, log_path: str = "none", level: int = logging.WARNING):

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

    # Create file handler
    # if log_path != "none":
    #     create_dir(os.path.split(log_path)[0])
    #
    #     file_handler = logging.FileHandler(log_path)
    #     file_handler.setLevel(level)
    #     file_handler.setFormatter(formatter)
    #     custom_logger.addHandler(file_handler)

    # if catch_output:
    #     catch_output_stream_to_logger(custom_logger)

    return custom_logger


class StreamToLogger(object):
    """
    Fake stream object that catches writes and writes them to the logger
    """

    def __init__(self, target_logger: logging.Logger, level: int):
        self.logger = target_logger
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


def catch_output_stream_to_logger(
        target_logger: logging.Logger, level: int = logging.INFO
):
    sys.stdout = StreamToLogger(target_logger, level)
    sys.stderr = StreamToLogger(target_logger, logging.ERROR)
