# pylint: disable=no-member

from dependency_injector import containers, providers
from src.utils.utils import Utils
from src.core.training.model_training import ModelTraining
from src.core.preprocessing.data_explorer import DataExplorer
from src.core.training.time_series_feature_engineering import TimeSeriesFeatureEngineering
from src.core.preprocessing.time_series_outlier_detector import TimeSeriesOutlierDetector


class Container(containers.DeclarativeContainer):
    """
    Dependency injection container for managing application services and components.

    Provides factories for creating instances of various application services.

    Attributes:
        utils (providers.Singleton): Singleton provider for the Utils class.
        model_training (providers.Factory): Factory provider for ModelTraining.
        data_explorer (providers.Factory): Factory provider for DataExplorer.
        time_series_feature_engineering (providers.Factory): Factory provider for TimeSeriesFeatureEngineering.
        time_series_outlier_detector (providers.Factory): Factory provider for TimeSeriesOutlierDetector.
    """
    utils = providers.Singleton(Utils)
    model_training = providers.Factory(ModelTraining, utils=utils)
    data_explorer = providers.Factory(DataExplorer, utils=utils)
    time_series_feature_engineering = providers.Factory(
        TimeSeriesFeatureEngineering)
    time_series_outlier_detector = providers.Factory(TimeSeriesOutlierDetector)
