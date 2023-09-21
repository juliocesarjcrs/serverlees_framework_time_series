from src.enums.file_type import FileType
from src.core.strategies.storage_strategy import StorageStrategy
from src.types.content_data import ContentData
from src.core.storage.local.local_file_reader import LocalFileReader
from src.core.storage.local.local_file_writer import LocalFileWriter


class LocalStrategy(StorageStrategy):

    def read(self, type_file: str, path: str, options=None):
        if type_file == FileType.MODEL.value:
            file_reader = LocalFileReader(path)
            return file_reader.load_model()
        elif type_file == FileType.CSV.value:
            file_reader = LocalFileReader(path)
            if options is None:
                options = {}
            return file_reader.read_csv(**options)
        else:
            raise ValueError('type_file do not configured')

    def save(self, type_file: FileType, content: ContentData):
        path = content['directory']
        file_name = content['file_name']
        file = content['file']
        if type_file == FileType.MODEL.value:
            file_writer = LocalFileWriter(path)
            return file_writer.save_model(file_name, file)
        elif type_file == FileType.CSV.value:
            file_writer = LocalFileWriter(path)
            return file_writer.save_dataframe_as_csv(file_name, file)
        else:
            raise ValueError('type_file do not configured')
