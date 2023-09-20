"""
Module for generating JSON responses.

This module defines the `JsonResponse` class, which contains a static method
`handler_json_response` for creating JSON responses with a specified status code and data.

Classes:
    JsonResponse: A class for generating JSON responses.

"""

import json

class JsonResponse:
    """
    A class for generating JSON responses.

    This class provides a static method, `handler_json_response`, for creating JSON responses
    with a specified status code and data.

    Attributes:
        None

    Methods:
        handler_json_response(data: any, status_code: int = 200) -> dict:
            Generates a JSON response with the provided data and status code.

    Example:
        response = JsonResponse.handler_json_response({"message": "Success"}, status_code=200)

    """

    @staticmethod
    def handler_json_response(data: any, status_code: int = 200) -> dict:
        """
        Generates a JSON response with the provided data and status code.

        Args:
            data (any): The data to be included in the response.
            status_code (int, optional): The HTTP status code for the response (default is 200).

        Returns:
            dict: A dictionary representing the JSON response.
                {
                    "statusCode": int,
                    "data": str (JSON-encoded data)
                }

        Example:
            response = JsonResponse.handler_json_response({"message": "Success"}, status_code=200)

        """
        return {
            "statusCode": status_code,
            "body": json.dumps(data)
        }
