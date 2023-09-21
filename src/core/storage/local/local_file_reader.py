"""
Module that contains the LocalFileReader class for reading local files, including models and CSV data.

This module defines the LocalFileReader class, which is a utility for reading and loading local files,
such as machine learning models and CSV data.

Classes:
    LocalFileReader: A utility class for reading local files.

Functions:
    None

Usage:
    Create an instance of LocalFileReader with the file path, and use its methods to load models or read CSV data.

Example:
    from your_module import LocalFileReader

    # Initialize a LocalFileReader with a file path
    reader = LocalFileReader('path_to_file')

    # Load a machine learning model from the file
    model = reader.load_model()

    # Read CSV data into a DataFrame
    df = reader.read_csv()
"""
import pandas as pd
import joblib


class LocalFileReader:
    """
    A utility class for reading local files, including models and CSV data.

    Args:
        file_path (str): The path to the local file.

    Attributes:
        file_path (str): The path to the local file.

    """

    def __init__(self, file_path: str):
        """
        Initializes a LocalFileReader with the specified file path.

        Args:
            file_path (str): The path to the local file.
        """
        self.file_path = file_path

    def load_model(self):
        """
        Load a machine learning model from the specified file.

        Returns:
            object: The loaded machine learning model.
        """
        return joblib.load(self.file_path)

    def read_csv(self, **kwargs) -> pd.DataFrame:
        """
        Read a CSV file into a pandas DataFrame.

        Args:
            **kwargs: Additional keyword arguments to pass to pd.read_csv.

        Returns:
            pd.DataFrame: A DataFrame containing the data from the CSV file.
        """
        return pd.read_csv(self.file_path, **kwargs)
