import numpy as np
import pandas as pd
from collections import Counter
class MyKNN:
    def __init__(self, k=3, p=2):
        """K-Nearest Neighbours

        Args:
            k (int, optional): choose the number of nearest neighbors. Defaults to 3.
            p (int, optional): order of the Minkowski distance. Defaults to 2.
        """
        self.k = k
        self.p = p
    def minkowski_distance(self, point1, point2):
        """
        Calculate the Minkowski distance between two points.

        Parameters:
        point1 (array-like): First point.
        point2 (array-like): Second point.
        p (int or float): The order of the norm.

        Returns:
        float: The Minkowski distance between the two points.
        """
        point1 = np.array(point1)
        point2 = np.array(point2)
        if (point1.shape != point2.shape) or (len(point1.shape) != 1):
            raise ValueError("Points must have the same dimensions and be one-dimensional.")
        if (self.p <= 0) or (not isinstance(self.p, (int, float))):
            raise ValueError("Order 'p' must be a positive integer or float.")
        return np.sum(np.abs(point1 - point2) ** self.p) ** (1 / self.p)

    def distances_from_point(self, data, point):
        """
        Calculate the Minkowski distances from a given point to all points in the dataset.

        Parameters:
        data (array-like or DataFrame): Dataset containing multiple points.
        point (array-like): The reference point.
        p (int or float): The order of the norm.

        Returns:
        np.ndarray: Array of distances from the reference point to each point in the dataset.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        point = np.array(point)
        if len(point.shape) != 1 or point.shape[0] != data.shape[1]:
            raise ValueError("Reference point must be one-dimensional and match the number of features in the dataset.")
        
        distances = data.apply(lambda row: self.minkowski_distance(row, point), axis=1)
        return distances

    def predict_the_label(self, distances, labels):
        """get majority lables

        Args:
            distances (pd.Series): distances from the point to all other points
            lables (pd.Series): labels corresponding to the points
            k (int): number of nearest neighbors to consider

        Returns:
            int: most common label among the k nearest neighbors
        """
        distances.sort_values(inplace=True)
        nearest_labels = labels.loc[distances.index[:self.k]]
        most_common = Counter(nearest_labels).most_common(1)
        return most_common[0][0]
    
    def fit(self, X, y):
        """fit the model

        Args:
            X (pd.DataFrame): features
            y (pd.Series): labels
        """