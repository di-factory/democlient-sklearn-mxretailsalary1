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
    
"""
    In this file, just define the needed classes to run the refered experiment.
    1.- Create or choose a class to define the desired datapipeline
    2.- Create an instance of the datapipeline class with the corresponding features
    3.- Create or choose a class to define the desired ML pipeline
    4.- Create an instance of the mlpipeline class with the corresponding hyperparams
    
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
class Pytorch_FFNN_Regressor_mine(nn.Module):
    def __init__(
        self, input_dim, output_dim: int = 1
    ):  # regresor always returns one output
        super().__init__()

        self.layer_i = nn.Linear(input_dim, 100)
        self.normalize = nn.LayerNorm(100)
        self.layerh1 = nn.Linear(100, 150)
        # self.drop1 = nn.Dropout(0.20)
        self.layerh2 = nn.Linear(100, 150)
        # self.drop2 = nn.Dropout(0.20)

        self.layerh3 = nn.Linear(150, 50)
        self.drop3 = nn.Dropout(0.10)

        self.layer_o = nn.Linear(50, output_dim)

        self.batch_size = 160
        self.lr = 0.01
        self.num_epochs = 60
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)

    def forward(self, x):
        y = nn.ReLU()(self.layer_i(x))
        y = self.normalize(y)
        h1 = F.relu(self.layerh1(y))
        h2 = F.relu(self.layerh2(y))
        y = self.drop3(h1 + h2)

        y = F.relu(self.layerh3(y))
        y = self.layer_o(y)

        return y


My_Pytorch_Regressor = {
    'input_layer':{
        'id':'input',
        'layer': nn.Linear(34, 100),
        #'dropout': None,
        #'normalize': None,
        'activation':nn.ReLU()
    },
    'hidden_layers':{
        'id':'hidden',
        'hlayers':[
            {
            'id': 'hidden_1',
            'layer': nn.Linear(100, 150),
            'dropout': nn.Dropout(0.2),
            'normalize': nn.LayerNorm(150),
            'activation': nn.ReLU(),
            },
        {
            'id': 'hidden_2',
            'layer': nn.Linear(150, 150),
            'dropout': nn.Dropout(0.2),
            'normalize': nn.LayerNorm(150),
            'activation': nn.ReLU(),
            },
        {
            'id': 'hidden_3',
            'layer': nn.Linear(150, 50),
            'dropout': nn.Dropout(0.2),
            'normalize': nn.LayerNorm(50),
            'activation': nn.ReLU(),
            },
        ],
    },
    'output_layer':{
        'id':'output',
        'layer': nn.Linear(50, 1),
        #'dropout': None,
        #'normalize': None,
        'activation':nn.ReLU()
    },
    'hyperparams':{
        'batch_size': 160,
        'lr': 0.005,
        'num_epochs':700,
        'loss_func': nn.MSELoss(),
        'optimizer': 'Adam',
    }    
}


mx_retail_salary1_mlpipeline = Pytorch_FFNN_Regressor(S_layers = My_Pytorch_Regressor)


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
            "shuffle": True
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
