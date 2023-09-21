"""
Module containing the BoostedHybridModel class.

This module defines the BoostedHybridModel class, which is used to create and train a hybrid model
by combining two different base models for prediction.
"""
from typing import Union
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor

class BoostedHybridModel:
    """
    A class representing a Boosted Hybrid Model that combines two different models for prediction.

    Args:
        model_1 (BaseEstimator, RegressorMixin): The first base model.
        model_2 (BaseEstimator, RegressorMixin): The second base model.

    Attributes:
        model_1 (BaseEstimator, RegressorMixin): The first base model.
        model_2 (BaseEstimator, RegressorMixin): The second base model.
        y_fit (pd.Series, None): The predicted values from the first model.
        y_resid (pd.Series, None): The residuals computed after fitting the first model.

    Methods:
        fit(x_1: pd.DataFrame, x_2: pd.DataFrame, target: pd.Series) -> "BoostedHybridModel":
            Fit the hybrid model to the data.
        predict(x_1: pd.DataFrame, x_2: pd.DataFrame) -> pd.Series:
            Make predictions using the hybrid model.
    """

    def __init__(self, model_1: Union[BaseEstimator, RegressorMixin], model_2: Union[BaseEstimator, RegressorMixin]):
        """
        Initialize a BoostedHybridModel with two base models.

        Args:
            model_1 (BaseEstimator, RegressorMixin): The first base model.
            model_2 (BaseEstimator, RegressorMixin): The second base model.
        """
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_fit = None
        self.y_resid = None

    def fit(self, x_1: pd.DataFrame, x_2: pd.DataFrame, target: pd.Series) -> "BoostedHybridModel":
        """
        Fit the BoostedHybridModel to the data.

        Args:
            x_1 (pd.DataFrame): The input features for the first base model.
            x_2 (pd.DataFrame): The input features for the second base model.
            target (pd.Series): The target values for training.

        Returns:
            BoostedHybridModel: The trained hybrid model.
        """
        self.model_1.fit(x_1, target)

        y_fit = pd.Series(
            self.model_1.predict(x_1),
            index=x_1.index
        )

        # Calculate residuals
        y_resid = target - y_fit

        # Fit the second model on residuals
        self.model_2.fit(x_2, y_resid)

        # Save the data
        self.y_fit = y_fit
        self.y_resid = y_resid

        # Return the trained hybrid model
        return self

    def predict(self, x_1: pd.DataFrame, x_2: pd.DataFrame) -> pd.Series:
        """
        Make predictions using the BoostedHybridModel.

        Args:
            x_1 (pd.DataFrame): The input features for the first base model.
            x_2 (pd.DataFrame): The input features for the second base model.

        Returns:
            pd.Series: The predicted values.
        """
        y_pred = pd.Series(
            self.model_1.predict(x_1),
            index=x_1.index
        )
        y_pred += self.model_2.predict(x_2)
        return y_pred
