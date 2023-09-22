"""
ContentData Module

This module defines the ContentData class, which is a data class for representing content information.

Example:
    content = ContentData(
        file='model_fit_data',
        directory='./buckets/expense-control-bucket/',
        file_name='best_model.pkl'
    )
"""

from dataclasses import dataclass

@dataclass
class ContentData:
    """
    Data class for representing content information.

    Attributes:
        file (str): The model fit information.
        directory (str): The directory where the file will be saved.
        file_name (str): The name of the file to be saved.
    """

    file: str
    directory: str
    file_name: str
