import json
import pandas as pd
from fastapi import APIRouter
from src.core.facades.data_processing_facade import DataProcessingFacade
from src.error.lambda_error_handler import LambdaErrorHandler
from src.core.responses.json_response import JsonResponse
from src.core.strategies.storage_context import StorageContext
from src.core.training.model_selector import ModelSelector
from src.enums.file_type import FileType
from src.types.content_data import ContentData


router = APIRouter()


@router.get("/preprocessing/generate-processed-data")
def generate_processed_data(type_storage: str, path_base: str):
    """ get result prediction

    Args:
        event (any): event lambda function

    Returns:
        _type_: _description_
    """
    anomalies = procesed_raw_data(type_storage, path_base)

    return {'anomalies': anomalies.to_dict(orient='records')}
def generate_processed_data_handler(query_sring_parameters: dict, body):
    """ get result prediction

    Args:
        event (any): event lambda function

    Returns:
        _type_: _description_
    """
    type_storage = query_sring_parameters['type_storage']
    path_base = query_sring_parameters['path_base']
    anomalies = procesed_raw_data(type_storage, path_base)
 

    return JsonResponse.handler_json_response({'anomalies': anomalies.to_dict(orient='records')})

def procesed_raw_data(type_storage, path_base):
    context = StorageContext(type_storage)
    options_to_save = {
        'parse_dates': ["date"],
        'usecols': [1, 5],
    }
    directory = f'{path_base}/data/raw/expenses.csv' if path_base else "data/raw/expenses.csv"
    expenses_df = context.read_file(
        FileType.CSV.value, directory , options_to_save)
    expenses_df['date'] = expenses_df['date'].dt.tz_localize(None)

    df_history = load_historical_data(type_storage, path_base)

    processed_df = pd.concat([expenses_df, df_history], ignore_index=True)
    processed_df.set_index('date', inplace=True)
    resampled_data = processed_df.resample('M').cost.sum()
    resampled_data.drop(resampled_data.tail(1).index, inplace=True)
    df_time = resampled_data.to_frame()
    directory_to_save = f'{path_base}/data/processed'if path_base else "data/processed"

    content: ContentData = {
        'file': df_time,
        'directory': directory_to_save,
        'file_name': 'df_time_monthly.csv'
    }
    options_to_save = {
        'index': True,
        'date_format': '%Y-%m-%d'
    }

    context.save_file(FileType.CSV.value, content, options_to_save)

    #---------------------Save witout outliers -----------------

    data_processing_facade = DataProcessingFacade(df_time)
    df_time_without_outlier, anomalies = data_processing_facade.get_data_without_outliers(
        'cost')
    anomalies.reset_index(inplace=True)
    anomalies['date'] = anomalies['date'].dt.strftime('%Y-%m-%d')
    content: ContentData = {
        'file': df_time_without_outlier,
        'directory': directory_to_save,
        'file_name': 'df_time_monthly_without_outliers.csv'
    }

    context.save_file(FileType.CSV.value, content, options_to_save)
    return anomalies

def load_historical_data(type_storage: str, path_base:str) -> pd.DataFrame:
    context = StorageContext(type_storage)
    directory_load_historical = f"{path_base}/data/raw/GASTOS-2019 - Flujo de Caja MES.csv" if path_base else "data/raw/GASTOS-2019 - Flujo de Caja MES.csv"
    file_name = directory_load_historical
    range1 = [i for i in range(3, 25)]
    options = {
        'index_col': None,
        'usecols': range1,
    }
    df_history = context.read_file(
        FileType.CSV.value, file_name, options)
    df_cut = df_history.iloc[[2, 8]]
    df_cut.columns = df_cut.iloc[0].values
    df_cut = df_cut.reset_index(drop=True)
    df_cut.drop(index=0, axis=0, inplace=True)
    df_cut = df_cut.melt(var_name='date', value_name='cost')
    df_cut['date'] = pd.to_datetime(df_cut['date'], format="%Y-%m-%d")
    df_cut['cost'] = df_cut['cost'].str.replace('$', '')
    df_cut['cost'] = df_cut['cost'].str.replace('.', '')
    df_cut['cost'] = df_cut['cost'].astype('int64')
    return df_cut


def handler(event: dict, context: dict):
    """
    handler

    Args:
        event (dict): El evento que desencadenó la función Lambda.
        context (dict): El contexto de ejecución de la función Lambda.

    Returns:
        _type_: _description_
    """
    try:
        query_sring_parameters = event['queryStringParameters']
        if not query_sring_parameters:
            raise ValueError("queryStringParameters undefined")
        body = event['body']
        return generate_processed_data_handler(query_sring_parameters, body)

    except Exception as exception:
        error_response = LambdaErrorHandler.handle_error(exception)
        return error_response
