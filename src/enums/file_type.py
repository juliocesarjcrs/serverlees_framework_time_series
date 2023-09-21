"""Module containing the FileType enumeration.

This module defines the FileType enumeration that represents available storage types.
"""

from enum import Enum

class FileType(Enum):
    """FIle Type

    Args:
        Enum (_type_): _description_
    """
    MODEL = 'MODEL'
    CSV = 'CSV'
