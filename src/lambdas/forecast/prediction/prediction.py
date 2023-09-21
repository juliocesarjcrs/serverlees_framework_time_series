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
    path_base = query_sring_parameters['path_base']
    context = StorageContext(type_storage)

    model_loaded = context.read_file(
        FileType.MODEL.value, f'{path_base}/best_model.pkl')

    prediction_facade = PredictionFacade(model_loaded)

    parameters_training = read_model_metrics(context, path_base)
    params_model = params_model = {'months_to_predict': months_to_predict,
                                   'train_columns': parameters_training}
    response_data = prediction_facade.predict(params_model)
    return JsonResponse.handler_json_response(response_data)


def read_model_metrics(context, path_base):
    # get metrics
    options = {
        'index_col': False,
    }
    model_metrics_df = context.read_file(
        FileType.CSV.value, f'{path_base}/model_metrics.csv', options)
    if "Unnamed: 0" in model_metrics_df.columns:
        model_metrics_df.drop(columns=["Unnamed: 0"], inplace=True)
    # Convertir 'Key' en índice para facilitar el acceso.
    return model_metrics_df.loc[2, 'train_columns']


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
