"""Module containing the StorageType enumeration.

This module defines the StorageType enumeration that represents available storage types.
"""

from enum import Enum

class StorageType(Enum):
    """Storage Type

    Args:
        Enum (_type_): _description_
    """
    S3_AWS = 'S3_AWS'
    LOCAL = 'LOCAL'
