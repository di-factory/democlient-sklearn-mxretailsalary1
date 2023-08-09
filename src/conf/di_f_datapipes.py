# version: july15, 2023

from src.conf.di_f_logging import di_f_logger

from omegaconf import DictConfig
import pandas as pd
import numpy as np
import os

from typing import Optional, Callable

import torch

from pycaret.internal.pipeline import Pipeline as Pycaret_Pipeline
from pycaret.internal.preprocess.preprocessor import (
    PowerTransformer,
    StandardScaler,
    SimpleImputer,
)
from pycaret.internal.preprocess.preprocessor import (
    FixImbalancer,
    TransformerWrapper,
    TargetEncoder,
    OneHotEncoder,
    MinMaxScaler,
)


from torch.utils.data import Dataset

import joblib


def load_data(pathfile: os.path, encoding: str = "") -> pd.DataFrame:
    """
    to load data from raw directory and return a dataframe
    """
    di_f_logger.info(f"load_data(): loading raw data...@ {pathfile}")

    df = pd.read_csv(pathfile, encoding=encoding)

    di_f_logger.info(f"        Were loaded: (rows,cols) {df.shape}")

    return df


def write_transformed(pathfile: os.path, data: pd.DataFrame) -> None:
    """
    saving transformed data in file
    """
    data.to_csv(pathfile, index=False)

    di_f_logger.info(f"write_transformed(): File were created @ {pathfile}")


def write_spplited(
    pathfile_train_features: os.path,
    pathfile_train_labels: os.path,
    pathfile_validation_features: os.path,
    pathfile_validation_labels: os.path,
    pathfile_test_features: os.path,
    pathfile_test_labels: os.path,
    train_data: pd.DataFrame,
    train_labels: pd.DataFrame,
    val_data: pd.DataFrame,
    val_labels: pd.DataFrame,
    test_data: pd.DataFrame,
    test_labels: pd.DataFrame,
) -> None:
    """
    splitting and writting six types of files: train futures & labels where trainning will be done;
    validation future & labels, where trainning validates. The % of records reserved to validation
    is defined with parameter percent_valid. And test future and labels records as unseen records to
    test the model before to migrate a production. The % of records reserved to test is defined with
    parameter percent_test.
    The logic about percent mangement is: Over the 100% of data available, percent_test is reserved as
    unseen data. Over the rest, percent_valid is reseved for validation.
    """

    train_data.to_csv(pathfile_train_features, index=None)
    train_labels.to_csv(pathfile_train_labels, index=None)

    val_data.to_csv(pathfile_validation_features, index=None)
    val_labels.to_csv(pathfile_validation_labels, index=None)

    test_data.to_csv(pathfile_test_features, index=None)
    test_labels.to_csv(pathfile_test_labels, index=None)

    di_f_logger.info(
        f"write_spplited(): Spplited data saved @ {pathfile_train_features}"
    )
    di_f_logger.info(f"                             {pathfile_train_labels}")
    di_f_logger.info(f"                             {pathfile_validation_features}")
    di_f_logger.info(f"                             {pathfile_validation_labels}")
    di_f_logger.info(f"                             {pathfile_test_features}")
    di_f_logger.info(f"                             {pathfile_test_labels}")


# Neutral Transformers:


def minmax(
    values,
):  # This util, transform columns of a matrix between min-max and returns also the min/max values used in each column
    minmax_cols = {}
    for col in range(values.shape[1]):
        min_val = min(values[:, col])
        max_val = max(values[:, col])
        std = (values[:, col] - min_val) / (max_val - min_val)
        values[:, col] = std * (max_val - min_val) + min_val
        minmax_cols[col] = {"min_val": min_val, "max_val": max_val}
    return minmax_cols, values


def normalize(
    values,
):  # This util, normalize columns of a matrix and returns also the mean/std values used in each column
    normalize_cols = {}
    for col in range(values.shape[1]):
        mean = values[:, col].mean()
        std = values[:, col].std()
        values[:, col] = (values[:, col] - mean) / std
        normalize_cols[col] = {"mean": mean, "max_val": std}
    return normalize_cols, values


class Di_F_Transform_Categorical_to_num:
    def __init__(self):
        self.metadata = []

    def fit(self, X, y=None):
        for col in range(X.shape[1]):
            dict = {}
            dict["col"] = col
            dict["value_range"] = {}
            for num, val in enumerate(np.unique(X[:, col])):
                dict["value_range"][val] = num
            self.metadata.append(dict)
        di_f_logger.info(f"fitting Categorical to Num Transformer {dict}")

    def transform(self, X):
        X_result = X.copy()  # Make a copy of the input to avoid modifying it directly
        for r in range(X.shape[0]):
            for c in self.metadata:
                X_result[r, c["col"]] = c["value_range"][X[r, c["col"]]]
        di_f_logger.info(
            f"fTransformed Categorical to Num Transformer shape result: {X_result.shape}"
        )
        return X_result


class DI_F_Transform_MinMax:
    def __init__(self):
        self.metadata = []

    def fit(self, X, y=None):
        for col in range(X.shape[1]):
            dict = {}
            dict["min"] = X[:, col].astype(float).min()
            dict["max"] = X[:, col].astype(float).max()
            self.metadata.append(dict)
        di_f_logger.info(f"fitting MinMax Transformer {dict}")

    def transform(self, X):
        X_result = X.astype(float)
        for col in range(X.shape[1]):
            X_result[:, col] = (X[:, col].astype(float) - self.metadata[col]["min"]) / (
                self.metadata[col]["max"] - self.metadata[col]["min"]
            )
        di_f_logger.info(
            f"fTransformed MinMax Transformer shape result: {X_result.shape}"
        )
        return X_result


class Di_F_Transform_Normalize:
    def __init__(self):
        self.metadata = []

    def fit(self, X, y=None):
        print(X)
        for col in range(X.shape[1]):
            dict = {}
            dict["mean"] = X[:, col].astype(float).mean()
            dict["std"] = X[:, col].astype(float).std()
            self.metadata.append(dict)
        di_f_logger.info(f"fitting Normalize Transformer {dict}")

    def transform(self, X):
        X_result = X.astype(float)
        for col in range(X.shape[1]):
            X_result[:, col] = (
                X[:, col].astype(float) - self.metadata[col]["mean"]
            ) / self.metadata[col]["std"]
        di_f_logger.info(
            f"fTransformed Normalize Transformer shape result: {X_result.shape}"
        )
        return X_result


class Di_F_Transform_OneHot:
    def __init__(self):
        self.metadata = []

    def fit(self, X, y=None):
        for col in range(X.shape[1]):
            dict = {}
            dict["col"] = col
            dict["value_range"] = {}
            for num, val in enumerate(np.unique(X[:, col])):
                dict["value_range"][val] = num
            self.metadata.append(dict)
        di_f_logger.info(f"fitting Onehot Transformer {dict}")

    def transform(self, X):
        num_columns = sum(len(col["value_range"]) for col in self.metadata)
        X_result = np.zeros((X.shape[0], num_columns))  # Initialize the result matrix

        current_col = 0
        for col_metadata in self.metadata:
            col = col_metadata["col"]
            col_mapping = col_metadata["value_range"]

            for r in range(X.shape[0]):
                value = X[r, col]
                value_index = col_mapping.get(
                    value, -1
                )  # Get the index from the mapping

                if value_index != -1:  # If value exists in mapping
                    X_result[
                        r, current_col + value_index
                    ] = 1  # Set the corresponding value to 1

            current_col += len(col_mapping)  # Move to the next set of columns
        di_f_logger.info(
            f"Transformed OneHot Transformer shape result: {X_result.shape}"
        )
        return X_result


#  Datapipelines clasess


class Di_F_Transformer:
    def __init__(self, transformation_map: list = None):
        self.txmap = transformation_map

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        for m in self.txmap:
            m["transformer"] = m["transformer"]()
            fields = m["fields"]
            m["transformer"].fit(X[:, fields])

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        for i, m in enumerate(self.txmap):
            transformed = m["transformer"].transform(X[:, m["fields"]])
            if not i:
                Xnew = transformed
            else:
                Xnew = np.hstack((Xnew, transformed))

        return pd.DataFrame(Xnew)


class PytorchDataSet(Dataset):
    def __init__(self, X, y, transforms: Optional[Callable] = None):
        self.X = X
        self.y = y
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Get the sample at the specified index
        input_sample = self.X[idx]
        label = self.y[idx]

        # Convert input and label to PyTorch tensors
        input_sample = torch.tensor(input_sample)
        label = torch.tensor(label)

        # Apply transformations if provided
        if self.transforms:
            for transform in self.transforms:
                input_sample, label = transform(input_sample, label)

        return input_sample, label


class Di_F_DataPipeline:
    def __init__(self, cfg: DictConfig, data_pipeline: Optional[Callable] = None):
        self.cfg = cfg
        self.dataPipeline = data_pipeline

    def fit(self, X, y):
        self.dataPipeline.fit(X, y)

    def transform(self, X):
        return self.dataPipeline.transform(X)

    def load_dataPipeline(
        self,
    ) -> (
        None
    ):  # this method loads the dataPipeline model that was created/saved in runDataPipeline()
        try:
            self.dataPipeline = joblib.load(
                os.path.join(
                    self.cfg.paths.models_dir, self.cfg.file_names.datapipeline
                )
            )
            di_f_logger.info("datapipeline loaded successfully!")
        except Exception as e:
            di_f_logger.info(f"Error loading the datapipeline: {e}")

    def save_dataPipeline(self) -> None:  # this method saves the dataPipeline model
        try:
            joblib.dump(
                self.dataPipeline,
                os.path.join(
                    self.cfg.paths.models_dir, self.cfg.file_names.datapipeline
                ),
            )
            di_f_logger.info("Datapipeline saved successfully!")
        except Exception as e:
            di_f_logger.info(f"Error saving the datapipeline: {e}")
