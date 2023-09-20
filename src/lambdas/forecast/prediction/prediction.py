from fastapi import FastAPI
from src.core.facades.prediction_facade import PredictionFacade
from src.error.lambda_error_handler import LambdaErrorHandler
from src.core.responses.json_response import JsonResponse
from src.core.strategies.storage_context import StorageContext
from src.enums.file_type import FileType

app = FastAPI()


@app.get("/predict")
def prediction(query_sring_parameters: dict):
    """ get result prediction

    Args:
        event (any): event lambda function

    Returns:
        _type_: _description_
    """
    type_storage = query_sring_parameters['type_storage']
    months_to_predict = query_sring_parameters['months_to_predict']

    context = StorageContext(type_storage)
    model_loaded = context.read_file(FileType.MODEL.value, 'buckets/local-bucket/models/best_model.pkl')
    prediction_facade = PredictionFacade(model_loaded)

    response_data = prediction_facade.predict(
        {'months_to_predict': months_to_predict})
    return JsonResponse.handler_json_response(response_data)

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
        return prediction(query_sring_parameters)

    except Exception as exception:
        error_response = LambdaErrorHandler.handle_error(exception)
        return error_response
