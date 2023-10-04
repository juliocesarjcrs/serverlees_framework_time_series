from enum import Enum

class AnomalyDetectionMethod(Enum):
    """
    Enum representing available anomaly detection methods.
    """

    DBSCAN = 'DBSCAN'
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based
    anomaly detection method.
    """

    Z_SCORE = 'Z Score'
    """
    The Z Score method is a statistical technique that measures the distance of a sample
    from the mean in terms of standard deviations.
    """
