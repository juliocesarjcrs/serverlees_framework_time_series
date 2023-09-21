from src.core.strategies.storage_strategy import StorageStrategy
from src.core.storage.s3Client.s3_client import S3Manager
from src.enums.file_type import FileType
from src.types.content_data import ContentData
from src.utils.logger.logger import Logger


class S3AwsStrategy(StorageStrategy):
    # logger = Logger("S3AwsStrategy")
    def read(self, type_file: FileType, path: str, options=None):
        # self.logger.nfo(f'::: Llegó type_file= {type_file} :::')
        if type_file == FileType.MODEL.value:
            s3_manager = S3Manager()
            object_data = s3_manager.load_model(
                "local-bucket", path)
            return object_data
        elif type_file == FileType.CSV.value:
            if options is None:
                options = {}
            s3_manager = S3Manager()
            return s3_manager.read_csv("local-bucket", path ** options)
        else:
            # self.logger.error(
            #     f'::: type_file do not configured = {type_file} :::')
            raise ValueError('S3AwsStrategy: type_file do not configured')

    def save(self, type_file: FileType, content: ContentData):
        path = content['directory']
        file_name = content['file_name']
        file = content['file']
        if type_file == FileType.MODEL.value:
            s3_manager = S3Manager()
            pass
        elif type_file == FileType.CSV.value:
            s3_manager = S3Manager()
            pass
        else:
            # self.logger.error(
            #     f'::: type_file do not configured = {type_file} :::')
            raise ValueError('S3AwsStrategy:type_file do not configured')
