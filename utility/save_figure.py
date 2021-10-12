import os
from utility.file_util import create_dir, file_exist
from utility.logger import get_logger


def save_figure(path: str, fig, overwrite: bool = False):
    parent, file = os.path.split(path)
    create_dir(parent, recursive=True)

    if not file_exist(path):
        fig.savefig(path)
        get_logger().info(f'Figure saved: {path}')
    elif file_exist(path) and overwrite:
        fig.savefig(path)
        get_logger().info(f'Figure overwritten: {path}')
    elif file_exist(path):
        raise FileExistsError()
