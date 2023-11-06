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
import sklearn.model_selection
import sklearn.pipeline
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")


class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        binary_columns = list(range(15))
        numerical_columns = list(range(15, 21))

        # data_train, data_test, target_train, target_test = \
        #     sklearn.model_selection.train_test_split(train.data,
        #                                              train.target,
        #                                              test_size=0.1,
        #                                              random_state=args.seed)
        
        column_transformer = sklearn.compose.ColumnTransformer([
            ("binary", sklearn.preprocessing.OneHotEncoder(categories="auto", sparse=False), binary_columns),
            ("numerical", sklearn.preprocessing.StandardScaler(), numerical_columns)
        ])

        model = sklearn.pipeline.Pipeline([
            ("preprocessing", column_transformer),
            ("poly_features", sklearn.preprocessing.PolynomialFeatures(degree=3, include_bias=False)),
            ("classifier", sklearn.linear_model.LogisticRegression())
        ])

        # TODO: Train a model on the given dataset and store it in `model`.
        # model.fit(data_train, target_train)
        # predictions = model.predict(data_test)
        # print(sklearn.metrics.accuracy_score(target_test, predictions))

        model.fit(train.data, train.target)


        # model.fit()

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


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
