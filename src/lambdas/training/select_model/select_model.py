import json
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from src.error.lambda_error_handler import LambdaErrorHandler
from src.core.responses.json_response import JsonResponse
from src.core.strategies.storage_context import StorageContext
from src.core.training.model_selector import ModelSelector
from src.enums.file_type import FileType
from src.types.content_data import ContentData
from src.types.bodys.selec_model_body import SelectModelBody
from src.core.training.model_training import ModelTraining
from src.enums.models_type import ModelsType
from src.core.training.time_series_feature_engineering import TimeSeriesFeatureEngineering
from src.utils.utils import Utils
from src.types.bodys.selec_model_body import datasetToTrain
router = APIRouter()


@router.post("/training/select-model")
async def select_model(request: Request, type_storage: str, path_base: str):
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
        raise HTTPException(
            status_code=400, detail=f"Error en el cuerpo de la solicitud {exception}")

    model_metrics, data_as_dict = process_models_datasets(
        type_storage, path_base, body)

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
    model_metrics, data_as_dict = process_models_datasets(
        type_storage, path_base, body)

    return JsonResponse.handler_json_response({'model_metrics': model_metrics, 'best_model': data_as_dict})


def process_models_datasets(type_storage: str, path_base: str, body: SelectModelBody):
    context = StorageContext(type_storage)

    train_models = body.train_models
    target_col = 'cost'
    utils = Utils()
    model_training = ModelTraining(utils, type_storage)
    # Set models
    for model_type in train_models:
        set_dynamic_models(model_training, model_type.value)

    # train an evaluate
    datasets_to_train = body.datasets_to_train
    establish_datasets(datasets_to_train, path_base,
                       target_col, context, model_training)

    # model_training.set_individual_dataset(processed_df, target_col)
    model_training.train_and_evaluate(target_col)
    # save metrics
    model_metrics = model_training.get_model_metrics()
   # ---------------------------------------------------------------------

    model_selector = ModelSelector()
    weights_dict = body.weights.model_dump()

    best_model_df = model_selector.analyze_model_metrics_summary(
        model_metrics, weights_dict)

    # Convertir el DataFrame en una lista de diccionarios
    data_as_dict = best_model_df.to_dict(orient='records')
    best_model_name = best_model_df.iloc[0]['model_name']
    best_dataset_name = best_model_df.iloc[0]['dataset_name']
    # ----------------------------------------------------------
    # save best model
    model_training.reset_dataset()
    model_training.reset_models()
    filter_datasets_to_train = filter(lambda x: (
        x.name.value == best_dataset_name), datasets_to_train)
    filter_datasets_to_train = list(filter_datasets_to_train)
    set_dynamic_models(model_training, best_model_name)
    establish_datasets(filter_datasets_to_train, path_base,
                       target_col, context, model_training)

    result_best_model = model_training.train_and_evaluate(target_col, True)
    model_fit = result_best_model['model_fit']
    directory_to_save = f'{path_base}/models'if path_base else "models"
    content: ContentData = {
        'file': model_fit,
        'directory': directory_to_save,
        'file_name': 'best_model.pkl'
    }

    context.save_file(FileType.MODEL.value, content)

    save_metrics_to_csv(context, result_best_model, path_base)
    return model_metrics, data_as_dict


def add_complete_features_enginiering(dataframe: pd.DataFrame):
    time_series_feature_engineering = TimeSeriesFeatureEngineering()
    processed_df, _ = time_series_feature_engineering.feature_engineering_time_series_dynamic(
        dataframe)
    return time_series_feature_engineering.add_monthly_holiday_count(processed_df, 'holidays_cont')


def establish_datasets(datasets_to_train: list[datasetToTrain], path_base: str, target_col: str, context, model_training):
    for dataset in datasets_to_train:
        name = dataset.name.value
        path_database = dataset.path

        options = {
            'parse_dates': True,
            'index_col': 'date'
        }
        file_name = f'{path_base}/{path_database}' if path_base else path_database
        dataframe = context.read_file(
            FileType.CSV.value, file_name, options)
        processed_df = add_complete_features_enginiering(dataframe)
        x_data = processed_df.drop(target_col, axis=1)
        y_data = processed_df[target_col]
        params = {'name': name, 'train_ratio': 0.8, 'X': x_data, 'y': y_data}
        model_training.set_dataset(params)


def save_metrics_to_csv(context, result_best_model: dict, path_base: str):
    result_best_model.pop('model_fit', None)

    # Guardar el diccionario modificado en un archivo CSV
    df_metrics = pd.DataFrame([result_best_model])
    directory_to_save = f'{path_base}/models'if path_base else "models"
    content: ContentData = {
        'file': df_metrics,
        'directory': directory_to_save,
        'file_name': 'model_metrics.csv'
    }
    options_to_save = {
        'index': False,
        'date_format': '%Y-%m-%d'
    }
    context.save_file(FileType.CSV.value, content, options_to_save)


def set_dynamic_models(model_training, model_type: str):
    if model_type == ModelsType.XGBOOST.value:
        model_training.set_model_xgboost()
    elif model_type == ModelsType.LINEAR_REGRESSION.value:
        model_training.set_model_linear_regression()
    elif model_type == ModelsType.BOOSTED_HIBRID.value:
        model_training.set_model_boosted_hybrid()
    elif model_type == ModelsType.RANDOM_FOREST_REGRESSOR.value:
        model_training.set_model_Random_forest_regressor()
    else:
        raise ValueError(
            f'model_type does not definided {model_type}')


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
