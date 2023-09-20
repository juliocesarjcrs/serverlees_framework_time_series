from abc import ABC, abstractmethod
from src.enums.file_type import FileType
from src.types.content_data import ContentData


class StorageStrategy(ABC):
    @abstractmethod
    def read(self, type_file: FileType, path: str):
        pass

    def save(self, type_file: FileType, content: ContentData):
        pass
