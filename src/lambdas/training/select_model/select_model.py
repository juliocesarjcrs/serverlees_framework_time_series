import json
from fastapi import FastAPI
from src.core.facades.data_processing_facade import DataProcessingFacade
from src.error.lambda_error_handler import LambdaErrorHandler
from src.core.responses.json_response import JsonResponse
from src.core.strategies.storage_context import StorageContext
from src.core.training.model_selector import ModelSelector
from src.enums.file_type import FileType
from src.types.content_data import ContentData


app = FastAPI()


@app.get("/training/select-model")
def select_model(query_sring_parameters: dict, body):
    """ get result prediction

    Args:
        event (any): event lambda function

    Returns:
        _type_: _description_
    """
    type_storage = query_sring_parameters['type_storage']
    context = StorageContext(type_storage)
    dataframe = context.read_file(
        FileType.DATAFRAME.value, 'buckets/local-bucket/data/processed/df_time_monthly_without_outliers.csv')
    data_processing_facade = DataProcessingFacade(dataframe)
    model_metrics = data_processing_facade.train_models_individual(
        'date', 'cost', 'M')

    model_selector = ModelSelector()

    body_dict = json.loads(body)
    weights = body_dict.get("weights", {})
    best_model_df = model_selector.analyze_model_metrics_summary(
        model_metrics, weights)
    # Convertir el DataFrame en una lista de diccionarios
    data_as_dict = best_model_df.to_dict(orient='records')
    best_model_name = best_model_df.iloc[0]['model_name']
    # save best model
    result_best_model = data_processing_facade.get_best_model(
        'cost', best_model_name)

    model_fit = result_best_model['model_fit']
    content: ContentData = {
        'file': model_fit,
        'directory': './buckets/local-bucket/models',
        'file_name': 'best_model.pkl'
    }

    context.save_file(FileType.MODEL.value, content)

    return JsonResponse.handler_json_response({'model_metrics': model_metrics, 'best_model': data_as_dict})


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
        return select_model(query_sring_parameters, body)

    except Exception as exception:
        error_response = LambdaErrorHandler.handle_error(exception)
        return error_response
