#!/usr/bin/env python3
# e97917d7-2509-11ec-986f-f39926f24a9c
# aecaa3bb-2101-11ec-986f-f39926f24a9c
# d989c517-2472-11ec-986f-f39926f24a9c

import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt

import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, Binarizer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline  
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import norm

# Transform similar to OneHot but instead of putting 1 on one value
# it puts a Gaussian distribution around the value
class GaussianOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, std_dev=0.5):
        self.std_dev = std_dev
        self.label_encoders_ = []
        self.n_values_ = []

    def fit(self, X, y=None):
        X = np.array(X)
        self.label_encoders_ = [sklearn.preprocessing.LabelEncoder().fit(col) for col in X.T]
        self.n_values_ = [len(le.classes_) for le in self.label_encoders_]
        return self

    def transform(self, X):
        X = np.array(X)
        X_transformed = [le.transform(col) for le, col in zip(self.label_encoders_, X.T)]
        one_hot_encoded = np.zeros((X.shape[0], sum(self.n_values_)))

        start = 0
        for i, n_values in enumerate(self.n_values_):
            end = start + n_values
            for j in range(n_values):
                mask = (X_transformed[i] == j).astype(float)
                
                # Create a circular Gaussian distribution
                x = np.arange(n_values)
                distance = np.minimum(np.abs(x - j), n_values - np.abs(x - j))
                gaussian_curve = norm.pdf(distance, loc=0, scale=self.std_dev)
                
                gaussian_curve /= gaussian_curve.sum()  # Normalize to ensure sum equals 1
                one_hot_encoded[:, start:end] += np.outer(mask, gaussian_curve)
            start = end

        return one_hot_encoded


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")


class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - 0: season (1: winter, 2: spring, 3: summer, 4: autumn)
    - 1: year (0: 2011, 1: 2012)
    - 2: month (1-12)
    - 3: hour (0-23)
    - 4: holiday (binary indicator)
    - 5: day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - 6: working day (binary indicator; a day is neither weekend nor holiday)
    - 7: weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - 8: temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - 9: feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - 10: relative humidity (0-1 range)
    - 11: windspeed (normalized to 0-1 range)

    The target variable is the number of rented bikes in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    # EncoderTest()

    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # data_train, data_test, target_train, target_test = \
        #     sklearn.model_selection.train_test_split(train.data,
        #                                              train.target,
        #                                              test_size=0.1,
        #                                              random_state=args.seed)
        
        # Column types, other columns are numerical
        categorical_columns = [0, 1, 2, 3, 5]
        binary_columns = [4, 6]
        ordinal_columns = [7]

        # applies transforms based on column types
        column_transformer = ColumnTransformer([
            ("circ_gaussian_one_hot", GaussianOneHotEncoder(std_dev=1.0), categorical_columns),
            # ("one_hot", OneHotEncoder(sparse=False, handle_unknown="ignore"), categorical_columns),
            ("binary", Binarizer(), binary_columns),
            ("ordinal", OrdinalEncoder(), ordinal_columns),
            ("scaler", StandardScaler(), [i for i in range(train.data.shape[1]) if i not in categorical_columns and i not in binary_columns and i not in ordinal_columns])
        ])

        # pipeline that applies transforms and then adds polynomial features and then trains a model
        model = Pipeline([
            ("preprocessing", column_transformer),
            ("poly_features", PolynomialFeatures(2, include_bias=False)),
            ("poisson_regressor", PoissonRegressor(alpha=0.2, max_iter=1000))
            # ("ridge_regressor", Ridge(alpha=0.1))
        ])
        
        # TRAINING ON ALL DATA FOR COMPETITION
        model.fit(train.data, train.target)
        
        # TRAINING ON TRAINING DATA FOR TESTING
        # model.fit(data_train, target_train)        
        # predictions = model.predict(data_test)
        # rmse = np.sqrt(sklearn.metrics.mean_squared_error(target_test, predictions))
        # print(rmse)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions
    
    
def EncoderTest():
    # Days of the week represented as numbers 1 (Monday) to 7 (Sunday)
    days_of_week = np.array([
        [1], [2], [3], [4], [5], [6], [7]
    ])

    # Creating the GaussianSmoothedOneHotEncoder with a small standard deviation
    encoder = GaussianOneHotEncoder(std_dev=0.5)

    # Fit and transform the data
    encoded_days = encoder.fit_transform(days_of_week)

    # Print the original and transformed data
    print("Original Days of the Week (1: Monday, 7: Sunday):")
    print(days_of_week.ravel())

    print("\nGaussian Smoothed One-Hot Encoded Days:")
    for day, encoded_day in zip(days_of_week, encoded_days):
        print(f"{int(day[0])}: {encoded_day}")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
