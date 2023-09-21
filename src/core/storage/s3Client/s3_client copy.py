import tempfile
import boto3
from botocore.config import Config
import joblib
import os
from src.utils.logger.logger import Logger
from botocore.exceptions import ClientError
from botocore.exceptions import NoCredentialsError

class S3Manager:
    """A class for managing interactions with AWS S3.

    This class provides methods for configuring an S3 client, loading models from S3, and more.

    Attributes:
        None
    """
    _instance = None
    logger = None

    def __new__(cls):
        # self.logger = Logger("S3Client")
        if cls._instance is None:
            cls._instance = super(S3Manager, cls).__new__(cls)
            cls._instance.client = boto3.client('s3')
            is_offline = os.environ['IS_OFFLINE']
            # self.logger.info(f'::: is_offline ::: {is_offline}')
            access_key = 'S3RVER' if is_offline else 'mykey'
            secret_key = 'S3RVER' if is_offline else 'mysecret'
            endpoint_url = 'http://localhost:4569' if is_offline else 'http://<minio_server_ip>:9000'

            cls._instance.client = boto3.client('s3',
                                        endpoint_url=endpoint_url,
                                        aws_access_key_id=access_key,
                                        aws_secret_access_key=secret_key,
                                        config=Config(signature_version='s3v4'))
        return cls._instance


    # def __init__(self, bucket_name: str):
    #     """
    #     :param bucket: A Boto3 Bucket resource. This is a high-level resource in Boto3
    #                     that wraps bucket actions in a class-like structure.
    #     """
    #     self.name_bucket = bucket_name
    #     self.s3_client = None

    # def get_s3_client(self):
    #     """Get an S3 client instance.

    #     Returns:
    #         botocore.client.S3: An S3 client instance configured with the specified settings.
    #     """
    #     is_offline = os.environ['IS_OFFLINE']
    #     print('is_offline', is_offline)
    #     access_key = 'S3RVER' if is_offline else 'mykey'
    #     secret_key = 'S3RVER' if is_offline else 'mysecret'
    #     endpoint_url = 'http://localhost:4569' if is_offline else 'http://<minio_server_ip>:9000'

    #     self.s3_client = boto3.client('s3',
    #                                   endpoint_url=endpoint_url,
    #                                   aws_access_key_id=access_key,
    #                                   aws_secret_access_key=secret_key,
    #                                   config=Config(signature_version='s3v4'))

    def load_model_from_s3(self, object_key: str):
        """Load a machine learning model from an S3 bucket.

        Args:
            bucket (str): The name of the S3 bucket.
            object_key (str): The key (path) to the model file in the S3 bucket.

        Returns:
            Any: The loaded machine learning model.

        Raises:
            Exception: If there is an error while loading the model.
        """
        try:
            self.get_s3_client()
            self.exists()
            # return self.read_object_from_bucket(object_key)
            self.list_objects_in_bucket()
            # self.list_files_in_folder('./buckets')
            return self.read_object_from_bucket(object_key)
        except Exception as exception:
            raise Exception("An error occurred: {}".format(exception))

    def exists(self):
        """
        Determine whether the bucket exists and you have access to it.

        :return: True when the bucket exists; otherwise, False.
        """
        try:
            self.s3_client.head_bucket(Bucket=self.name_bucket)
            self.logger.info(f"Bucket {self.name_bucket} exists.")
            exists = True
        except ClientError:
            self.logger.warning(
                f"Bucket %s doesn't exist or you don't have access to it. {self.name_bucket}")
            exists = False
        return exists

    def read_object_from_bucket(self, object_key: str):
        """
        Reads an object from an S3 bucket.

        Args:
            object_key (str): The key of the object to be read from the bucket.

        Returns:
            bytes: The data of the read object in bytes format.

        Raises:
            Exception: If an error occurs while reading the object.
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.name_bucket, Key=object_key)
            object_data = response['Body'].read()
            return object_data
        except Exception as exception:
            raise Exception("An error occurred: {}".format(exception))

    def list_objects_in_bucket(self):
        """
        Lists objects in an S3 bucket.

        Returns:
            list: A list of object keys (filenames) in the bucket.

        Raises:
            Exception: If an error occurs while listing objects.
        """
        try:
            response = self.s3_client.list_objects(Bucket=self.name_bucket)
            object_keys = [obj['Key'] for obj in response.get('Contents', [])]
            return object_keys
        except Exception as exception:
            raise Exception(
                "An error occurred while listing objects: {}".format(exception))
