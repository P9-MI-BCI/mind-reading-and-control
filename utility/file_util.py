import os


def create_dir(path: str, recursive=False):
    """
    Create a new folder if it does not exist
    By setting recursive to True the function will create parent folders also
    """
    parent, folder = os.path.split(path)

    if not os.path.exists(parent):
        if recursive:
            create_dir(parent, recursive=recursive)
        else:
            raise FileNotFoundError(
                f"Parent directory not found: {parent} you can run the function recursively by setting recursive=True"
            )

    if not os.path.exists(path):
        os.makedirs(path)