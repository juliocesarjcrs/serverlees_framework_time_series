import tempfile
import io
import os
import sys
import boto3
import joblib
import pandas as pd
from io import StringIO, BytesIO
from botocore.exceptions import ClientError
from src.utils.logger.logger import Logger


class S3Manager:
    _instance = None
    s3_client = None  # Declaración del atributo client
    logger = Logger("S3Manager")

    def __init__(self):
        self.bucket_name = 'expense-control-bucket'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(S3Manager, cls).__new__(cls)
            cls.s3_client = cls._create_s3_client()
        return cls._instance

    @staticmethod
    def _create_s3_client():
        is_offline = os.environ.get('IS_OFFLINE')
        if is_offline:
            access_key = 'S3RVER'
            secret_key = 'S3RVER'
            endpoint_url = 'http://localhost:4569'
            return boto3.client('s3'
                                # endpoint_url=endpoint_url,
                                # aws_access_key_id=access_key,
                                # aws_secret_access_key=secret_key
                                )
        else:
            aws_access_key_id = ''
            aws_secret_access_key = ''
            return boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    def exists(self, bucket_name):
        """
        Determine whether the bucket exists and you have access to it.

        Args:
            bucket_name (str): El nombre del bucket a verificar.

        Returns:
            bool: True si el bucket existe y tienes acceso a él; de lo contrario, False.
        """
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
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
            response = self.s3_client.get_object(
                Bucket=self.bucket_name, Key=object_key)
            object_data = response['Body'].read()
            return object_data
        except Exception as exception:
            raise Exception("An error occurred: {}".format(exception))

    def load_model(self, object_key: str):
        try:
            self.logger.info(
                f'::: STEP 1 Load model object_key_path= {object_key}:::')

            # Descargar el modelo desde S3
            response = self.s3_client.get_object(
                Bucket=self.bucket_name, Key=object_key)
            model_binary = response['Body'].read()

            # Cargar el modelo desde el objeto binario
            loaded_model = joblib.load(BytesIO(model_binary))

            return loaded_model
        except Exception as exception:
            self.logger.error(f"Error reading the model from S3: {exception}")
            raise Exception("An error occurred: {}".format(exception))

    # def load_model(self, object_key: str):
    #     try:
    #         temp_file_path = os.path.join(tempfile.gettempdir(), object_key)
    #         self.logger.info(
    #             f'::: STEP 1 Load model object_key_path= {object_key}:::')

    #         # Descargar el modelo desde S3 a un archivo temporal

    #         self.client.download_file(
    #             self.bucket_name, object_key, temp_file_path)

    #         self.logger.info(
    #             f'::: STEP 2 Download temp_file_path {temp_file_path}:::')
    #         # self.list_files_in_tmp()
    #         # self.list_installed_libraries()
    #         # Cargar el modelo desde el archivo temporal
    #         loaded_model = joblib.load(temp_file_path)
    #         return loaded_model

        # except Exception as exception:
        #     self.logger.error(f"Error reading the model from S3: {exception}")
        #     raise Exception("An error occurred: {}".format(exception))

    def list_files_in_tmp(self):
        tmp_dir = '/tmp'  # Directorio temporal en Lambda

        # Listar los archivos en el directorio /tmp
        file_list = os.listdir(tmp_dir)

        # Imprimir la lista de archivos
        for file_name in file_list:
            self.logger.info(file_name)

    def read_csv(self, object_key: str, **kwargs):
        try:
            self.logger.info(f"::: object key ::: {object_key}")
            response = self.s3_client.get_object(
                Bucket=self.bucket_name, Key=object_key)
            csv_bytes = response['Body'].read()

            csv_str = csv_bytes.decode('utf-8')
            dataframe = pd.read_csv(io.StringIO(csv_str), **kwargs)
            return dataframe
        except ClientError as exception:
            self.logger.error(
                f"Error reading the CSV file from S3:: {exception}")
            raise Exception("An error occurred: {}".format(exception))
        except Exception as exception:
            self.logger.error(
                f"Error reading the CSV file from S3:: {exception}")
            raise Exception("An error occurred: {}".format(exception))

    def save_model(self, object_key: str, directory: str,  model):
        try:
            # Especifica un nombre de archivo para guardar el modelo en /tmp
            model_filename = f"/tmp/{object_key}"

            # Guarda el modelo en /tmp
            joblib.dump(model, model_filename)

            # Lee el archivo recién guardado en bytes
            with open(model_filename, 'rb') as model_file:
                model_bytes = model_file.read()
            file_path_to_save = os.path.join(directory, object_key)
            # Sube el archivo al bucket de S3
            self.logger.info(f"::: SAVE: object key  ::: {file_path_to_save}")
            self.s3_client.put_object(
                Bucket=self.bucket_name, Key=file_path_to_save, Body=model_bytes)

            self.logger.info(
                f"Model saved to S3: s3://{self.bucket_name}/{object_key}")

            # Limpia el archivo temporal en /tmp
            os.remove(model_filename)

        except ClientError as exception:
            self.logger.error(f"Error saving the model to S3: {exception}")
            raise Exception("An error occurred: {}".format(exception))

    def list_installed_libraries(self):
        installed_libraries = sys.modules.keys()
        for library in installed_libraries:
            self.logger.info(library)

    def save_dataframe_as_csv(self, object_key: str, dataframe: pd.DataFrame, **kwargs):
        try:
            self.logger.info(f"::: SAVE: object key  ::: {object_key}")
            # Convertir el DataFrame en una cadena CSV en memoria
            csv_buffer = StringIO()
            dataframe.to_csv(csv_buffer, **kwargs)
            csv_content = csv_buffer.getvalue()
            self.s3_client.put_object(
                Bucket=self.bucket_name, Key=object_key, Body=csv_content.encode())

            self.logger.info(
                f"Dataframe to Csv saved to S3: s3://{self.bucket_name}/{object_key}")
        except Exception as exception:
            self.logger.error(
                f"Error saving Dataframe to Csv to S3: {exception}")
            raise Exception("An error occurred: {}".format(exception))

    def save_binary_data_as_csv(self, object_key: str, binary_data: bytes, **kwargs):
        try:
            # El contenido CSV en forma de cadena
            csv_content = binary_data.decode('utf-8')

            # Convierte la cadena en una secuencia de bytes en formato UTF-8
            csv_bytes = csv_content.encode('utf-8')
            self.logger.info(f"::: object_key ::: {object_key}")
            self.s3_client.put_object(
                Bucket=self.bucket_name, Key=object_key, Body=csv_bytes)

            self.logger.info(
                f"Data Csv saved to S3: s3://{self.bucket_name}/{object_key}")
        except ClientError as exception:
            self.logger.error(
                f"Error ClientError the model to S3: {exception}")
            raise Exception("An error occurred: {}".format(exception))
        except Exception as exception:
            self.logger.error(
                f"Error Exception saving the model to S3: {exception}")
            raise Exception("An error occurred: {}".format(exception))
