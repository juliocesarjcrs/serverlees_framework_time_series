import pandas as pd
from src.enums.file_type import FileType
from src.core.strategies.storage_strategy import StorageStrategy
from src.types.content_data import ContentData
from src.core.storage.local.local_file_reader import LocalFileReader
from src.core.storage.local.local_file_writer import LocalFileWriter


class LocalStrategy(StorageStrategy):

    def read(self, type_file: FileType, path: str, options=None):
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

    def save(self, type_file: FileType, content: ContentData, options=None):
        if options is None:
            options = {}
        path = content['directory']
        file_name = content['file_name']
        file_send = content['file']
        file_writer = LocalFileWriter(path)
        if type_file == FileType.MODEL.value:
            return file_writer.save_model(file_name, file_send)
        elif type_file == FileType.CSV.value:
            if isinstance(file_send, pd.DataFrame):
                return file_writer.save_dataframe_as_csv(file_name, file_send, **options)
            elif  isinstance(file_send, bytes):
                return file_writer.save_binary_data_as_csv(file_name, file_send, **options)
            else:
                raise ValueError('content is not pd.Dataframe or binary')
        elif type_file == FileType.HTML.value:
            file_writer.save_graph_as_html(file_name, file_send)

        else:
            raise ValueError('type_file do not configured')
