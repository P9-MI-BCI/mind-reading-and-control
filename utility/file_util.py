import os
from utility.logger import get_logger


def create_dir(path: str, recursive=False):
    """
    Create a new folder if it does not exist
    By setting recursive to True the function will create parent folders also
    """
    parent, folder = os.path.split(path)

    if not os.path.exists(parent):
        if recursive:
            create_dir(parent, recursive=recursive)
            get_logger().info(f'Created directory: {parent} recursively')
        else:
            raise FileNotFoundError(
                get_logger().debug(
                    f'Parent directory not found: {parent} you can run the function recursively by setting '
                    f'recursive=True')
            )

    if not os.path.exists(path):
        os.makedirs(path)
        get_logger().info(f'Created directory: {path}')


def file_exist(path: str) -> bool:
    if not os.path.exists(path):
        return False
    else:
        return True
