"""
ModelSelector Module

This module contains the ModelSelector class, which is used for analyzing model metrics summaries and calculating weighted metrics.

Example:
    weights = {'RMSE': 0, 'MAE': 0, 'R2': 0, 'MAPE': 0, 'SMAPE': 1}
    metrics = [
        {
            'model_name': 'XGBoost',
            'metrics': {
                'MSE': 251508219658.9,
                'MAE': 386472.3,
                'R2': 0.4067758590909575,
                'RMSE': 501505.9517681719,
                'MAPE': 0.11792691047331545,
                'SMAPE': 11.620657619242802
            }
        },
        {
            'model_name': 'LinearRegression',
            'metrics': {
                'MSE': 655263808562.7455,
                'MAE': 713219.6396391153,
                'R2': -0.5455491292913164,
                'RMSE': 809483.6678789421,
                'MAPE': 0.23238534167107514,
                'SMAPE': 20.41681339101246
            }
        }
    ]

    analyzer = ModelSelector()
    summary = analyzer.analyze_model_metrics_summary(metrics, weights)
"""
from typing import List, Dict, Union
import pandas as pd

class ModelSelector:
    """
    ModelSelector is a class for analyzing model metrics summaries and calculating weighted metrics.

    Attributes:
        None

    Methods:
        analyze_model_metrics_summary(metrics: List[Dict[str, Union[float, Dict[str, float]]]], weights: Dict[str, int]) -> pd.DataFrame:
            Analyze a summary of model metrics and calculate the weighted metric.

    Example:
        weights = {'RMSE': 0, 'MAE': 0, 'R2': 0, 'MAPE': 0, 'SMAPE': 1}
        metrics = [
            {
                'model_name': 'XGBoost',
                'metrics': {
                    'MSE': 251508219658.9,
                    'MAE': 386472.3,
                    'R2': 0.4067758590909575,
                    'RMSE': 501505.9517681719,
                    'MAPE': 0.11792691047331545,
                    'SMAPE': 11.620657619242802
                }
            },
            {
                'model_name': 'LinearRegression',
                'metrics': {
                    'MSE': 655263808562.7455,
                    'MAE': 713219.6396391153,
                    'R2': -0.5455491292913164,
                    'RMSE': 809483.6678789421,
                    'MAPE': 0.23238534167107514,
                    'SMAPE': 20.41681339101246
                }
            }
        ]

        analyzer = ModelSelector()
        summary = analyzer.analyze_model_metrics_summary(metrics, weights)
    """

    def analyze_model_metrics_summary(self, metrics: List[Dict[str, Union[float, Dict[str, float]]]], weights: Dict[str, int]) -> pd.DataFrame:
        """
        Analyze a summary of model metrics and calculate the weighted metric.

        Args:
            metrics (List[Dict[str, Union[float, Dict[str, float]]]]): A list of dictionaries containing model metrics information.
                Each dictionary should have the following keys:
                - 'model_name': The name of the model.
                - 'metrics': A dictionary with specific metrics such as 'RMSE', 'MAE', 'R2', 'MAPE', and 'SMAPE'.
            weights (Dict[str, int]): A dictionary containing weights for the metrics. Keys should match the metrics ('RMSE', 'MAE', 'R2', 'MAPE', and 'SMAPE').

        Returns:
            pd.DataFrame: A DataFrame containing the summary of weighted metrics for each model.
                The DataFrame has the following columns:
                - 'model_name': The name of the model.
                - 'weighted_metric': The calculated weighted metric based on the provided weights.

        Example:
            weights = {'RMSE': 0, 'MAE': 0, 'R2': 0, 'MAPE': 0, 'SMAPE': 1}
            metrics = [
                {
                    'model_name': 'XGBoost',
                    'metrics': {
                        'MSE': 251508219658.9,
                        'MAE': 386472.3,
                        'R2': 0.4067758590909575,
                        'RMSE': 501505.9517681719,
                        'MAPE': 0.11792691047331545,
                        'SMAPE': 11.620657619242802
                    }
                },
                {
                    'model_name': 'LinearRegression',
                    'metrics': {
                        'MSE': 655263808562.7455,
                        'MAE': 713219.6396391153,
                        'R2': -0.5455491292913164,
                        'RMSE': 809483.6678789421,
                        'MAPE': 0.23238534167107514,
                        'SMAPE': 20.41681339101246
                    }
                }
            ]

            analyzer = ModelSelector()
            summary = analyzer.analyze_model_metrics_summary(metrics, weights)
        """
        # Create a DataFrame from the list of metrics
        results_df = pd.DataFrame(metrics)

        # Extract individual metrics from each metrics dictionary
        results_df['RMSE'] = results_df['metrics'].apply(lambda x: x.get('RMSE', 0))
        results_df['MAE'] = results_df['metrics'].apply(lambda x: x.get('MAE', 0))
        results_df['R2'] = results_df['metrics'].apply(lambda x: x.get('R2', 0))
        results_df['MAPE'] = results_df['metrics'].apply(lambda x: x.get('MAPE', 0))
        results_df['SMAPE'] = results_df['metrics'].apply(lambda x: x.get('SMAPE', 0))

        # Apply weights to the metrics
        results_df['weighted_metric'] = (
            weights['RMSE'] * results_df['RMSE'] +
            weights['MAE'] * results_df['MAE'] +
            weights['R2'] * results_df['R2'] +
            weights['MAPE'] * results_df['MAPE'] +
            weights['SMAPE'] * results_df['SMAPE']
        )

        # Group by 'model_name' and calculate the mean of weighted metrics
        summary_df = results_df.groupby(['model_name']).agg(
            {'weighted_metric': 'mean'}).reset_index()

        # Sort the DataFrame by weighted metric in ascending order
        summary_df = summary_df.sort_values('weighted_metric', ascending=True)

        # Display the DataFrame with metrics
        return summary_df