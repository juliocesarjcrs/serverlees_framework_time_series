import pandas as pd
import os
from src.core.strategies.storage_strategy import StorageStrategy
from src.core.storage.s3Client.s3_manager import S3Manager
from src.enums.file_type import FileType
from src.types.content_data import ContentData
from src.utils.logger.logger import Logger


class S3AwsStrategy(StorageStrategy):
    logger = Logger("S3AwsStrategy")

    def read(self, type_file: FileType, path: str, options=None):
        try:
            self.logger.info(f'::: Llegó type_file= {type_file} :::')
            if type_file == FileType.MODEL.value:
                s3_manager = S3Manager()
                object_data = s3_manager.load_model(
                    path)
                return object_data
            elif type_file == FileType.CSV.value:
                if options is None:
                    options = {}
                s3_manager = S3Manager()
                self.logger.info(
                    f'::: Va a función read_csv options= {options} :::')
                return s3_manager.read_csv(path, ** options)
            else:
                # self.logger.error(
                #     f'::: type_file do not configured = {type_file} :::')
                raise ValueError('S3AwsStrategy: type_file do not configured')
        except Exception as exception:
            self.logger.error(f'::: exception = {exception} :::')

    def save(self, type_file: FileType, content: ContentData, options=None):
        directory = content['directory']
        file_name = content['file_name']
        file_send = content['file']
        file_path_to_save = os.path.join(directory, file_name)
        if type_file == FileType.MODEL.value:
            s3_manager = S3Manager()
            s3_manager.save_model(file_name, directory, file_send)
        elif type_file == FileType.CSV.value:
            s3_manager = S3Manager()
            if isinstance(file_send, pd.DataFrame):
                s3_manager.save_dataframe_as_csv(file_path_to_save, file_send, **options)
            elif isinstance(file_send, bytes):
                s3_manager.save_binary_data_as_csv(file_path_to_save, file_send, **options)
            else:
                raise ValueError('content is not pd.Dataframe or binary')
        else:
            # self.logger.error(
            #     f'::: type_file do not configured = {type_file} :::')
            raise ValueError('S3AwsStrategy:type_file do not configured')
