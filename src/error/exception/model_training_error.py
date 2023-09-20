class ModelTrainingError(Exception):
    """Excepción personalizada para errores durante el entrenamiento de modelos."""

    def __init__(self, model_name, message):
        self.model_name = model_name
        self.message = message
        super().__init__(self.message)