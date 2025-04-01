import logging
import os
from typing import Tuple


class FileUtil:
    """Utility class for handling files."""

    @staticmethod
    def file_exists(path: str) -> Tuple[bool, str, str]:
        """
        Checks if the specified file exists.

        Args:
            path (str): The file path.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        file_name = os.path.basename(path)
        file_name = os.path.splitext(file_name)[0]
        if os.path.isfile(path):
            logging.info(msg=f"File exists: {path}")
            return True, path, file_name
        else:
            logging.info(msg=f"File does not exist: {path}")
            return False, path, file_name
