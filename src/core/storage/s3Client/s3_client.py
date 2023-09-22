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

    def __init__(self):
        self.bucket_name = 'expense-control-bucket'


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
            response = self.client.get_object(
                Bucket=self.bucket_name, Key=object_key)
            object_data = response['Body'].read()
            return object_data
        except Exception as exception:
            raise Exception("An error occurred: {}".format(exception))

    def load_model(self, object_key: str):
        try:
            response = self.client.get_object(
                Bucket=self.bucket_name, Key=object_key)
            model_bytes = response['Body'].read()
            loaded_model = joblib.load(model_bytes)
            return loaded_model
        except ClientError as exception:
            self.logger.error(f"Error reading the model from S3: {exception}")
            raise Exception("An error occurred: {}".format(exception))

    def read_csv(self, object_key: str, **kwargs):
        try:
            response = self.client.get_object(
                Bucket=self.bucket_name, Key=object_key)
            csv_bytes = response['Body'].read()
            csv_str = csv_bytes.decode('utf-8')
            dataframe = pd.read_csv(io.StringIO(csv_str), **kwargs)
            return dataframe
        except ClientError as exception:
            self.logger.error(
                f"Error reading the CSV file from S3:: {exception}")
            raise Exception("An error occurred: {}".format(exception))

    def save_model(self, object_key: str, model):
        try:
            # Especifica un nombre de archivo para guardar el modelo en /tmp
            model_filename = f"/tmp/{object_key}"

            # Guarda el modelo en /tmp
            joblib.dump(model, model_filename)

            # Lee el archivo recién guardado en bytes
            with open(model_filename, 'rb') as model_file:
                model_bytes = model_file.read()

            # Sube el archivo al bucket de S3
            self.client.put_object(
                Bucket=self.bucket_name, Key=object_key, Body=model_bytes)

            self.logger.info(
                f"Model saved to S3: s3://{self.bucket_name}/{object_key}")

            # Limpia el archivo temporal en /tmp
            os.remove(model_filename)

        except ClientError as exception:
            self.logger.error(f"Error saving the model to S3: {exception}")
            raise Exception("An error occurred: {}".format(exception))
