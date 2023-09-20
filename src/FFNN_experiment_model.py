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
import torch.nn.functional as F
from collections import OrderedDict

from src.conf.di_f_mlpipes import Pytorch_FFNN_Regressor, Pytorch_Optimizers
from src.conf.di_f_datapipes import (
    Di_F_DataPipeline,
    Di_F_Transform_OneHot,
    DI_F_Transform_MinMax,
    Di_F_Transformer,
)

from src.conf.di_f_logging import di_f_logger

"""
    In this file, just define the needed classes to run the refered experiment.
    1.- Create or choose a class to define the desired datapipeline
    2.- Create an instance of the datapipeline class with the corresponding features
    3.- Create or choose a class to define the desired ML pipeline
    4.- Create an instance of the mlpipeline class with the corresponding hyperparams
    5.- Create the final class level=3 to wrap datapipeline and mlpiepline
"""

#  1.- Create or choose a class to define the desired datapipeline

#  2.- Create an instance of the datapipeline class with the corresponding features
mx_retail_salary1_datapipeline = Di_F_Transformer(
    [
        {"id": "one-hot", "transformer": Di_F_Transform_OneHot, "fields": [0]},
        {"id": "MinMax", "transformer": DI_F_Transform_MinMax, "fields": [1, 2]},
    ]
)


#  3.- Create or choose a class to define the desired ML pipeline
#  3.a Case: Whole class particularly created for this:
class Pytorch_FFNN_Regressor_mine(nn.Module):
    def __init__(
        self, input_dim, output_dim: int = 1
    ):  # regresor always returns one output
        super().__init__()

        self.layers = (
            nn.ModuleDict()
        )  # to write the different layers of the pytorch regression model
        self.hyperparams: dict = {}  # to write the hyperparams of the model

        self.layers["input"] = nn.Linear(input_dim, 100)
        self.layers["input_normalize"] = nn.LayerNorm(100)
        self.layers["hidden1"] = nn.Linear(100, 150)
        # self.layers['hidden1_drop'] = nn.Dropout(0.20)
        self.layers["hidden2"] = nn.Linear(100, 150)
        # self.layers['hidden2_drop'] = nn.Dropout(0.20)

        self.layers["hidden3"] = nn.Linear(150, 50)
        self.layers["hidden3_drop"] = nn.Dropout(0.20)

        self.layers["output"] = nn.Linear(50, output_dim)

        self.hyperparams["batch_size"] = 160
        self.hyperparams["lr"] = 0.01
        self.hyperparams["num_epochs"] = 60
        self.hyperparams["loss_func"] = nn.MSELoss()
        self.hyperparams["optimizer"] = torch.optim.Adam(
            params=self.parameters(), lr=self.hyperparams["lr"]
        )

        di_f_logger.info(
            f"model definition: {self.layers} -- Hyperparams: {self.hyperparams}"
        )

    def forward(self, x):
        y = nn.ReLU()(self.layers["input"](x))
        y = self.layers["input_normalize"](y)
        h1 = F.relu(self.layers["hidden1"](y))
        h2 = F.relu(self.layers["hidden2"](y))
        y = self.layers["hidden3_drop"](h1 + h2)

        y = F.relu(self.layers["hidden3"](y))
        y = self.layers["output"](y)

        return y


#  3.b Case: Using predefined Pytorch class and use params in layers + hyperparams
#  with this following structure to define the pytorch regressor class to be used by the
#  Pytorch_FFNN_Regressor defined in di_f_mlpipes.py
My_Pytorch_Regressor = {
    "input_layer": {
        "id": "input",
        "layer": nn.Linear(34, 100),
        #'dropout': None,
        #'normalize': None,
        "activation": nn.ReLU(),
    },
    "hidden_layers": {
        "id": "hidden",
        "hlayers": [
            {
                "id": "hidden_1",
                "layer": nn.Linear(100, 150),
                "dropout": nn.Dropout(0.2),
                "normalize": nn.LayerNorm(150),
                "activation": nn.ReLU(),
            },
            {
                "id": "hidden_2",
                "layer": nn.Linear(150, 150),
                "dropout": nn.Dropout(0.2),
                "normalize": nn.LayerNorm(150),
                "activation": nn.ReLU(),
            },
            {
                "id": "hidden_3",
                "layer": nn.Linear(150, 50),
                "dropout": nn.Dropout(0.2),
                "normalize": nn.LayerNorm(50),
                "activation": nn.ReLU(),
            },
        ],
    },
    "output_layer": {
        "id": "output",
        "layer": nn.Linear(50, 1),
        #'dropout': None,
        #'normalize': None,
        "activation": nn.ReLU(),
    },
    "hyperparams": {
        "batch_size": 160,
        "lr": 0.005,
        "num_epochs": 700,
        "loss_func": nn.MSELoss(),
        "optimizer": "Adam",
    },
}

#  4.- Create an instance of the mlpipeline class with the corresponding hyperparams
# mx_retail_salary1_mlpipeline = Pytorch_FFNN_Regressor(S_layers = My_Pytorch_Regressor)
mx_retail_salary1_mlpipeline = Pytorch_FFNN_Regressor_mine(34)


#  5.- Create the final class level=3 to wrap datapipeline and mlpiepline
class MxRetailSalary1(Di_F_Pipe_Regression_Pytorch_FFNN):
    def __init__(
        self,
        cfg: DictConfig,
        data_pipeline=mx_retail_salary1_datapipeline,
        ml_pipeline=mx_retail_salary1_mlpipeline,
    ):
        super().__init__(cfg)

        #  Here goes the choosen metrics for the model
        self.scores = [
            {"id": "mape", "metric": mean_absolute_percentage_error},
            {"id": "r2", "metric": r2_score},
        ]

        #  And here the corresponding params for kfold fit method:
        self.kfold = {"n_splits": 5, "shuffle": True}

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
