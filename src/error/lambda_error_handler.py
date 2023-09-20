"""
Module to provide error handling for Lambda functions.

This module defines the `LambdaErrorHandler` class, which contains a static method
`handle_error` for handling errors in Lambda functions and returning JSON responses.

Classes:
    LambdaErrorHandler: A class for handling errors in Lambda functions.

"""
import json
import traceback
from src.utils.logger.logger import Logger


class LambdaErrorHandler:
    """
    A class for handling errors in Lambda functions and returning JSON responses.

    This class provides a static method, `handle_error`, that takes an exception as input,
    logs the error, and returns a JSON response containing error information.

    Attributes:
        None

    Methods:
        handle_error(exception: Exception) -> dict:
            Handles errors in a Lambda function and returns a JSON response.

    Example:
        error_response = LambdaErrorHandler.handle_error(ValueError("Invalid input"))

    """
    @staticmethod
    def handle_error(exception):
        """
        Handles errors in a Lambda function and returns a JSON response.

        Args:
            exception (Exception): The exception to be handled.

        Returns:
            dict: A dictionary with error information in JSON format.
                {
                    "error_type": str,
                    "error_message": str
                }

        Example:
            LambdaErrorHandler.handle_error(ValueError("Invalid input"))

        The `handle_error` method takes an exception as input, logs the error, and returns a JSON
        response containing the error type and message along with a 400 status code.

        """
        logger = Logger("LambdaErrorHandler")
        logger.error(exception)
         # Obtener la ubicación del error (archivo y línea)
        error_location = traceback.format_exc().splitlines()[-1]
        error_info = {
            "error_type": type(exception).__name__,
            "error_message": str(exception),
            "error_location": error_location
        }
        return {
            "statusCode": 400,
            'body': json.dumps({'error': error_info})
        }
