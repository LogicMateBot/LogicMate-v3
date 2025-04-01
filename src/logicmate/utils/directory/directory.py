import logging
import os


class DirectoryUtil:
    """Utility class for handling directories."""

    @staticmethod
    def directory_exists(path: str) -> bool:
        """
        Checks if the specified directory exists.

        Args:
            path (str): The directory path.

        Returns:
            bool: True if the directory exists, False otherwise.
        """
        return os.path.isdir(s=path)

    @staticmethod
    def create_directory(path: str) -> None:
        """
        Creates a directory at the specified path.
        If the directory already exists, no exception is raised.

        Args:
            path (str): The directory path.
        """
        try:
            os.makedirs(name=path, exist_ok=True)
            logging.info(msg=f"Directory created: {path}")
        except Exception as e:
            logging.info(msg=f"Error creating directory {path}: {e}")

    @staticmethod
    def ensure_directory(path: str) -> str:
        """
        Ensures that the directory exists. If it does not, the directory is created.

        Args:
            path (str): The directory path.

        Return:
            str: Directory path.
        """
        if not DirectoryUtil.directory_exists(path):
            DirectoryUtil.create_directory(path)
            return path
        else:
            logging.info(msg=f"Directory already exists: {path}")
            return path
