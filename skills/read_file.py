import os

def read_file(file_path: str) -> str | None:
    """
    Reads the text of a file given its path, after checking if it exists.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The content of the file, or None if the file doesn't exist.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return f.read()
    else:
        return None
