# model definition for the experimeent

from omegaconf import DictConfig

from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import HuberRegressor, BayesianRidge, Ridge

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

from src.conf.di_f_pipeline import Di_F_Pipe_Regression_Pytorch_FFNN
import src.conf.preprocessors as pp

from catboost import CatBoostRegressor
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import r2_score, mean_absolute_percentage_error

import torch
import torch.nn as nn
from collections import OrderedDict

from src.conf.di_f_mlpipes import Pytorch_FFNN_Regressor
from src.conf.di_f_datapipes import (
    Di_F_DataPipeline,
    Pytorch_Transformer,
    Pytorch_MinMax_Tx,
    Pytorch_OneHot_Tx,
)


# Create an instance of Pytorch Pipeline class:
mx_retail_salary1_datapipeline = Pytorch_Transformer(
    [
        {"id": "one-hot", "transformer": Pytorch_OneHot_Tx, "fields": [0]},
        {"id": "Min", "transformer": Pytorch_MinMax_Tx, "fields": [1, 2]},
    ]
)
# ... and an instance of model Pycaret Pipeline:
mx_retail_salary1_mlpipeline = Pytorch_FFNN_Regressor(input_dim=34)


class MxRetailSalary1(Di_F_Pipe_Regression_Pytorch_FFNN):
    class Features(
        BaseModel
    ):  # Rewritting Features class to include the actual features
        state: str = "Hidalgo"
        income_employee_day: float = 4000.00
        employees_business: int = 6

    def __init__(
        self,
        cfg: DictConfig,
        data_pipeline=mx_retail_salary1_datapipeline,
        ml_pipeline=mx_retail_salary1_mlpipeline,
    ):
        super().__init__(cfg)

        self.scores = [
            {"id": "mape", "metric": mean_absolute_percentage_error},
            {"id": "r2", "metric": r2_score},
        ]

        self.kfold = {
            "n_splits": 5,
            "shuffle": True,
            "random_state": self.cfg.general_ml.seed,
        }

        # here you define the datapipeline transformation model getting params from pycaret in data profiling (notebook)
        self.dataPipeline = Di_F_DataPipeline(cfg, data_pipeline)

        # and here you define the prediction model
        self.model = ml_pipeline

    def runDataPipeline(self) -> dict:
        result = super().runDataPipeline()
        return result

    def fit(self, tracking: bool) -> dict:
        result = super().fit(tracking)
        return result

    def fit_Kfold(self, tracking: bool = False) -> dict:
        result = super().fit_Kfold(tracking)
        return result

    def predict(self, X: pd.DataFrame):
        return super().predict(X)
