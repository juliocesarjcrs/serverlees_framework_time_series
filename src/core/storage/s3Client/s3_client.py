import boto3
import os
import io
import joblib
import pandas as pd
from botocore.exceptions import ClientError
from src.utils.logger.logger import Logger


class S3Manager:
    _instance = None
    client = None  # Declaración del atributo client
    logger = Logger("S3Manager")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(S3Manager, cls).__new__(cls)
            cls.client = cls._create_s3_client()
        return cls._instance

    @staticmethod
    def _create_s3_client():
        is_offline = os.environ.get('IS_OFFLINE')
        if is_offline:
            access_key = 'S3RVER'
            secret_key = 'S3RVER'
            endpoint_url = 'http://localhost:4569'
        else:
            access_key = 'mykey'
            secret_key = 'mysecret'
            endpoint_url = 'http://<minio_server_ip>:9000'  # Reemplaza con la dirección real

        return boto3.client('s3',
                            endpoint_url=endpoint_url,
                            aws_access_key_id=access_key,
                            aws_secret_access_key=secret_key,
                            config=boto3.session.Config(signature_version='s3v4'))

    def exists(self, bucket_name):
        """
        Determine whether the bucket exists and you have access to it.

        Args:
            bucket_name (str): El nombre del bucket a verificar.

        Returns:
            bool: True si el bucket existe y tienes acceso a él; de lo contrario, False.
        """
        try:
            self.client.head_bucket(Bucket=bucket_name)
            return True
        except ClientError as exception:
            if exception.response['Error']['Code'] == '403':
                # No tienes acceso al bucket
                return False
            elif exception.response['Error']['Code'] == '404':
                # El bucket no existe
                return False
            else:
                # Ocurrió un error inesperado
                raise exception

    def read_object_from_bucket(self, bucket_name: str, object_key: str):
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
            response = self.client.get_object(
                Bucket=bucket_name, Key=object_key)
            object_data = response['Body'].read()
            return object_data
        except Exception as exception:
            raise Exception("An error occurred: {}".format(exception))

    def load_model(self, bucket_name: str, object_key: str):
        try:
            response = self.client.get_object(
                Bucket=bucket_name, Key=object_key)
            model_bytes = response['Body'].read()
            loaded_model = joblib.load(model_bytes)
            return loaded_model
        except ClientError as exception:
            self.logger.error(f"Error reading the model from S3: {exception}")
            raise Exception("An error occurred: {}".format(exception))

    def read_csv(self, bucket_name: str, object_key: str, **kwargs):
       try:
            response = self.client.get_object(Bucket=bucket_name, Key=object_key)
            csv_bytes = response['Body'].read()
            csv_str = csv_bytes.decode('utf-8')
            dataframe = pd.read_csv(io.StringIO(csv_str), **kwargs)
            return dataframe
       except ClientError as exception:
            self.logger.error(f"Error reading the CSV file from S3:: {exception}")
            raise Exception("An error occurred: {}".format(exception))