# here we define the different models
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List

from pycaret.internal.pipeline import Pipeline as Pycaret_pipeline
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor


Voting_Pycaret_model = Pycaret_pipeline(
    steps=[
        (
            "actual_estimator",
            VotingRegressor(
                estimators=[
                    (any),
                ],
                n_jobs=-1,
                weights=[],
            ),
        )
    ],
    verbose=False,
    memory=None,
)


# Optimizers for deep learning
class Di_F_Optimizers:
    subclass: str = None
    list: List = []

    def __init__(self):
        pass


class Pytorch_Optimizers(Di_F_Optimizers):
    list = [
        {"id": "Adam", "optim": torch.optim.Adam},
        {"id": "SGD", "optim": torch.optim.SGD},
    ]

    def __init__(self, id: str = "SGD"):
        self.subclass = "Pytorch"
        self.optimizer = None

        for opt_dupla in self.list:
            if id == opt_dupla["id"]:
                self.optimizer = opt_dupla["optim"]


class Pytorch_FFNN_Regressor(nn.Module):
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
        self.optimizer = Pytorch_Optimizers("Adam").optimizer(
            self.parameters(), lr=self.lr
        )

    def forward(self, x):
        y = F.relu(self.layer_i(x))
        y = self.normalize(y)
        h1 = F.relu(self.layerh1(y))
        h2 = F.relu(self.layerh2(y))
        y = self.drop3(h1 + h2)

        y = F.relu(self.layerh3(y))
        y = self.layer_o(y)

        return y


class Pytorch_FFNN(nn.Module):
    def __init__(self, nImputs: int, nOutputs: int, hidden: int, nHidden: int, dr=0.0):
        super().__init__()

        # Create a dictionary to save the layers
        self.layers = nn.ModuleDict()
        self.nImputs = nImputs  # Number of inputs as integer
        self.nOutputs = nOutputs  # Number of outputs
        self.nHidden = nHidden  # Number of hidden nodes by layer
        self.hidden = hidden  # Number of hidden layers
        self.dr = dr  # Dropuot rate

        self.losses = []  # For storing the losses at trainning

        ### input layer
        self.layers["input"] = nn.Linear(nImputs, nHidden)

        for h in range(self.hidden):
            ### hidden layer
            self.layers[f"hidden{h}"] = nn.Linear(nHidden, nHidden)

        ### output layer
        self.layers["output"] = nn.Linear(nHidden, nOutputs)

    def forward(self, x):
        pass
