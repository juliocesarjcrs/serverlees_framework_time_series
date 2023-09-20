from src.core.strategies.storage_strategy import StorageStrategy
from src.core.storage.s3Client.s3_client import S3Manager

class S3AwsStrategy(StorageStrategy):
    def read(self, type_file: str, path: str):
        if type_file == 'model':
            s3_manager = S3Manager()
            object_data = s3_manager.read_object_from_bucket("local-bucket", "mi-objeto")
            return object_data
        else:
            raise ValueError('type_file do not configured')