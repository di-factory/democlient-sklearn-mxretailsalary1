# here we define the different models
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


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


class Pytorch_FFNN_Regressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.layeri = nn.Linear(input_dim, 100)
        self.layerh1 = nn.Linear(100, 250)
        self.layerh2 = nn.Linear(250, 100)
        self.layero = nn.Linear(100, 1)

        self.batch_size = 200
        self.lr = 0.001
        self.num_epochs = 2000
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):
        y = F.relu(self.layeri(x))
        y = F.relu(self.layerh1(y))
        y = F.relu(self.layerh2(y))
        y = self.layero(y)

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
