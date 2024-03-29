from enum import Enum


class ModelsType(Enum):
    ARIMA = 'Arima'
    SARIMA = 'SARIMA'
    SARIMA_AUTO_ARIMA = 'SARIMA-AutoArima'
    XGBOOST = 'XGBoost'
    LINEAR_REGRESSION = 'LinearRegression'
    BOOSTED_HIBRID = 'BoostedHybrid'
    RANDOM_FOREST_REGRESSOR = 'RandomForestRegressor'
