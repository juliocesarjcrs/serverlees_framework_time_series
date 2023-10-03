
import requests
from fastapi import APIRouter, HTTPException, Request
from src.error.lambda_error_handler import LambdaErrorHandler
from src.core.responses.json_response import JsonResponse
from src.core.strategies.storage_context import StorageContext
from src.enums.file_type import FileType
from src.types.content_data import ContentData


router = APIRouter()


@router.post("/preprocessing/last-data")
async def get_last_data(request: Request, type_storage: str, path_base: str):
    """ get result prediction

    Args:
        event (any): event lambda function

    Returns:
        _type_: _description_
    """
    try:
        data = await request.json()
        body = data
        data_acces = body['data_acces']
        url_ext = body['url_ext']
    except Exception as exception:
        raise HTTPException(
            status_code=400, detail=f"Error en el cuerpo de la solicitud {exception}")

    access_token = getExternalAccesToken(url_ext, data_acces)
    getLastExpense(access_token, url_ext,  type_storage, path_base)

    return {'save': type_storage}


def get_last_data_handler(query_sring_parameters: dict, body):
    """ get result prediction

    Args:
        event (any): event lambda function

    Returns:
        _type_: _description_
    """
    type_storage = query_sring_parameters['type_storage']
    path_base = query_sring_parameters['path_base']
    url_ext = ''
    data_acces = ''
    access_token = getExternalAccesToken(url_ext, data_acces)
    getLastExpense(access_token, url_ext,  type_storage, path_base)

    return JsonResponse.handler_json_response({'save': type_storage})


def getLastExpense(access_token: str, url_ext: str, type_storage: str, path_base: str):
    url = f'{url_ext}/expenses/last/download'
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    response = requests.get(url, headers=headers)
    context = StorageContext(type_storage)
    if response.status_code == 200:
        directory = f"./{path_base}/data/raw" if path_base else "data/raw"
        content: ContentData = {
            'file': response.content,
            'directory': directory,
            'file_name': 'expenses.csv'
        }
        options_to_save = {
            'index': False,
            'date_format': '%Y-%m-%d'
        }
        context.save_file(FileType.CSV.value, content, options_to_save)

    else:
        raise ValueError(
            'Error en la solicitud. Código de estado:', response.status_code)


def getExternalAccesToken(url_ext, data_access):
    url = f'{url_ext}/auth/login'

    # Datos en formato JSON
    data = {
        "email": data_access['email'],
        "password": data_access['password']
    }

    # Cabeceras de la solicitud
    headers = {
        'Content-Type': 'application/json'
    }

    # Realizar la solicitud POST
    response = requests.post(url, json=data, headers=headers)

    # Verificar la respuesta
    if response.status_code == 200:
        response_json = response.json()

        # Acceder a los valores del JSON
        access_token = response_json.get('access_token')

        # Imprimir los valores obtenidos

        return access_token
    else:
        raise ValueError(
            'Error en la solicitud. Código de estado:', response.status_code)


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
        return get_last_data_handler(query_sring_parameters, body)

    except Exception as exception:
        error_response = LambdaErrorHandler.handle_error(exception)
        return error_response
