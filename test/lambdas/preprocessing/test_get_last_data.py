
from src.core.strategies.storage_context import StorageContext
from src.main import app
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# Crea un cliente de prueba para la aplicación
client = TestClient(app)


def nottest_preprocessing_last_data():
    # Datos de prueba
    request_data = {
        "data_acces": {
            "email": "test@coreo.com",
            "password": "password"
        },
        "url_ext": "http://url-external"
    }

    # Crear un mock para la clase de sesión de requests
    mock_session = Mock()

    # Configurar el comportamiento del mock para la función de requests.Session
    responses = [
        Mock(status_code=200, json=lambda: {
             "access_token": "mock_access_token"}),
        Mock(status_code=200, content=b"File content here")
    ]
    mock_session.post.side_effect = responses

    # Configurar el comportamiento para el segundo llamado (GET)
    mock_session.get.return_value.status_code = 200

    mock_response_get = Mock()
    mock_response_get.status_code = 200
    csv_content = (
        "id,cost,category,subcategory,created at,date\n"
        "42226,6800,Transporte,Transporte a clase inglés,2023-10-10T22:58:57.164Z,2023-10-10T00:00:00.000Z\n"
        "42225,5300,Transporte,Transporte a clase inglés,2023-10-10T17:01:14.475Z,2023-10-10T00:00:00.000Z\n"
        "42224,1080380,Cultura, diversión y esparcimiento,Vacaciones,2023-10-10T01:18:13.116Z,2023-10-09T00:00:00.000Z\n"
        "42223,9400,Salud,Medicamentos,2023-10-10T00:51:05.226Z,2023-10-09T00:00:00.000Z\n"
        "42222,6000,Aliment"
    )
    mock_response_get.content = csv_content.encode('utf-8')
    mock_session.get.return_value = mock_response_get

    # Crear un mock para StorageContext
    mock_context = Mock(spec=StorageContext)

    with patch('main.requests.Session', return_value=mock_session), \
            patch('src.core.strategies.storage_context.StorageContext', return_value=mock_context):
        # Realiza la solicitud al endpoint de preprocesamiento
        response = client.post(
            "/preprocessing/last-data?type_storage=LOCAL&path_base=buckets%2Fexpense-control-bucket", json=request_data)

    # Verifica que la respuesta tenga un código de estado 200 OK
    assert response.status_code == 200
    # assert response.json() == {"save": "LOCAL"}

    # En lugar de crear la excepción con el código de estado simulado, usa el código de estado real de la respuesta
    # if response.status_code != 200:
    #     raise ValueError(f'Error en la solicitud getLastExpense. Código de estado: {response.status_code}')
    # Verifica que la respuesta JSON sea un diccionario
