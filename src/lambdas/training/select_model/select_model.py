import json
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from src.core.facades.data_processing_facade import DataProcessingFacade
from src.error.lambda_error_handler import LambdaErrorHandler
from src.core.responses.json_response import JsonResponse
from src.core.strategies.storage_context import StorageContext
from src.core.training.model_selector import ModelSelector
from src.enums.file_type import FileType
from src.types.content_data import ContentData
from src.types.bodys.selec_model_body import SelectModelBody

router = APIRouter()

@router.post("/training/select-model")
async def select_model(request: Request, type_storage: str, path_base: str, csv_name:str):
    """ get result prediction

    Args:
        event (any): event lambda function

    Returns:
        _type_: _description_
    """
    try:
        data = await request.json()
        body = SelectModelBody(**data)
    except Exception as exception:
        raise HTTPException(status_code=400, detail="Error en el cuerpo de la solicitud")


    model_metrics , data_as_dict = get_best_model(type_storage, path_base, csv_name, body)

    return {'model_metrics': model_metrics, 'best_model': data_as_dict}

def select_model_handler(query_sring_parameters: dict, body):
    """ get result prediction

    Args:
        event (any): event lambda function

    Returns:
        _type_: _description_
    """
    type_storage = query_sring_parameters['type_storage']
    path_base = query_sring_parameters['path_base']
    csv_name = query_sring_parameters['csv_name']
    model_metrics , data_as_dict = get_best_model(type_storage, path_base, csv_name, body)

    return JsonResponse.handler_json_response({'model_metrics': model_metrics, 'best_model': data_as_dict})

def get_best_model(type_storage: str, path_base: str, csv_name: str, body: SelectModelBody):
    context = StorageContext(type_storage)

    options = {
        'parse_dates': True,
        'index_col': 'date',
    }
    dataframe = context.read_file(
        FileType.CSV.value, f'{path_base}/{csv_name}', options)
    data_processing_facade = DataProcessingFacade(dataframe)
    train_models = body.train_models
    model_metrics = data_processing_facade.train_models_individual(
        'date', 'cost', 'M', train_models)

    model_selector = ModelSelector()
    # body_dict = json.loads(body)
    # weights = body_dict.get("weights", {})
    # weights = body.weights.dict()
    weights_dict = body.weights.model_dump()
    best_model_df = model_selector.analyze_model_metrics_summary(
        model_metrics, weights_dict)
    # Convertir el DataFrame en una lista de diccionarios
    data_as_dict = best_model_df.to_dict(orient='records')
    best_model_name = best_model_df.iloc[0]['model_name']
    # save best model
    result_best_model = data_processing_facade.get_best_model(
        'cost', best_model_name)

    model_fit = result_best_model['model_fit']
    content: ContentData = {
        'file': model_fit,
        'directory': f'{path_base}/models',
        'file_name': 'best_model.pkl'
    }

    context.save_file(FileType.MODEL.value, content)

    save_metrics_to_csv(context, result_best_model, path_base)
    return model_metrics, data_as_dict

def save_metrics_to_csv(context, result_best_model: dict, path_base:str):
    result_best_model_without_model_fit = {
        key: value for key, value in result_best_model.items() if key != 'model_fit'}
    metrics_df = pd.DataFrame(result_best_model_without_model_fit)
    df_transposed = metrics_df.transpose()

    # Reiniciar el índice para que 'model_name' se convierta en una columna
    df_transposed.reset_index(inplace=True)

    # Cambiar el nombre de la columna 'index' a 'model_name' (opcional)
    df_transposed.rename(columns={'index': 'model_name'}, inplace=True)
    content: ContentData = {
        'file': df_transposed,
        'directory': f'./{path_base}/models',
        'file_name': 'model_metrics.csv'
    }
    context.save_file(FileType.CSV.value, content)


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
        return select_model_handler(query_sring_parameters, body)

    except Exception as exception:
        error_response = LambdaErrorHandler.handle_error(exception)
        return error_response
