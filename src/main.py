from fastapi import FastAPI
from src.lambdas.forecast.prediction import prediction
from src.lambdas.preprocessing.generate_processed_data import generate_processed_data
from src.lambdas.training.select_model import select_model

app = FastAPI()

# Monta las rutas definidas en los otros archivos
app.include_router(prediction.router)
app.include_router(generate_processed_data.router)
app.include_router(select_model.router)
