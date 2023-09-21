"""
Module that contains the StorageContext class.

This module defines the StorageContext class, which is used for handling storage strategies.
"""
from src.enums.file_type import FileType
from src.core.strategies.s3_aws_strategy import S3AwsStrategy
from src.core.strategies.local_strategy import LocalStrategy
from src.enums.storage_type import StorageType
from src.types.content_data import ContentData


class StorageContext:
    """
    A context class for handling storage strategies.

    Args:
        type_storage (StorageType): The type of storage to use (S3_AWS or LOCAL).

    Raises:
        ValueError: If type_storage is not a valid StorageType.

    Attributes:
        strategy: The storage strategy to be used.
    """

    def __init__(self, type_storage: StorageType):
        """
        Initializes a StorageContext with the specified type_storage.

        Args:
            type_storage (StorageType): The type of storage to use (S3_AWS or LOCAL).
        """
        # if type_storage == StorageType.S3_AWS.value:
        #     self.strategy = S3AwsStrategy()
        if type_storage == StorageType.LOCAL.value:
            self.strategy = LocalStrategy()
        else:
            raise ValueError("Invalid type_storage")


    def read_file(self, type_file: FileType, path: str, options= None):
        """
        Executes the selected storage strategy with additional arguments.

        Args:
            type_file (str): The type of file being operated on (e.g., 'dataframe', 'text', etc.).
            path (str): The file path or location for the storage strategy.

        Returns:
            str: The result of executing the storage strategy.
        """
        if options is None:
                options = {}
        return self.strategy.read(type_file, path, options=options)

    def save_file(self, type_file: FileType, content: ContentData):
        """
        Saves a file to the selected storage strategy.

        Args:
            type_file (str): The type of file being saved (e.g., 'dataframe', 'text', etc.).
            content (str): The content of the file to be saved.
            path (str): The file path or location for saving the file.

        Returns:
            str: A message indicating the success or failure of the save operation.
        """
        return self.strategy.save(type_file, content)
