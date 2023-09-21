import os
import joblib
import pandas as pd
from src.utils.logger.logger import Logger


class LocalFileWriter:
    """
    A class for saving and loading files locally.

    Args:
        directory (str): The directory where files will be saved and loaded from.

    Attributes:
        logger (Logger): An instance of the Logger class for logging messages.
        directory (str): The directory path for file operations.
    """

    def __init__(self, directory: str):
        """
        Initializes a LocalFileWriter instance with the specified directory.

        Args:
            directory (str): The directory where files will be saved and loaded from.
        """
        self.logger = Logger("LocalFileWriter")
        self.directory = directory
        self.validate_or_create_directory()

    def save_model(self, file_name: str, cls):
        """
        Saves a model object to a file in the specified directory.

        Args:
            file_name (str): The name of the file to save.
            cls: The model object to be saved.

        Returns:
            object: The loaded model object.
        """
        file_path = os.path.join(self.directory, file_name)
        joblib.dump(cls, file_path)
        save_sucessfull = joblib.load(file_path)
        self.logger.info(f'::: save_sucessfull ::: ${save_sucessfull}')

    def validate_or_create_directory(self):
        """
        Validates if the directory exists, and creates it if it doesn't.
        """
        isdir = os.path.isdir(self.directory)

        if not isdir:
            self.logger.info(
                f'::: Directory does not exist, it will be created ::: {self.directory}')
            os.mkdir(self.directory)

    def save_dataframe_as_csv(self, file_name: str, dataframe: pd.DataFrame):
        """
        Save a DataFrame to a CSV file while preserving the index and date frequency.

        Args:
            df (pandas.DataFrame): The DataFrame to be saved.
            directory (str): The directory path where the CSV file will be saved.
            file_name (str): The name of the CSV file.

        Returns:
            None
        """
        file_path_to_save = os.path.join(self.directory, file_name)
        dataframe.to_csv(file_path_to_save, index=True, date_format='%Y-%m-%d')
