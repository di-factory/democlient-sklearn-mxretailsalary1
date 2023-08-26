# here we define the different models
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List

from pycaret.internal.pipeline import Pipeline as Pycaret_pipeline
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
import matplotlib.pyplot as plt

from src.conf.di_f_logging import di_f_logger

#  pytorch base models
"""Di_F_NN = {
    'nImputs': int = 1, #  Number of input features
    'nOutputs': int = 1, #  Number of output features
    'hiddenLayers': List = [], # Hidden layer composition
}"""

# utilities
def plot_losses(train_losses, val_losses, pathfile):
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig(pathfile)
    plt.show()


# Pycaret Regression Model:
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

Layer: List = {'id':'', 'layer':None, 'dropout':None, 'normalize':None, 'activation':None}

NN_layers = {
    'input_layer': Layer,
    'hidden_layers': [Layer],
    'output_layers': Layer
}


#  Pytorch FNN regression Model
class Pytorch_FFNN_Regressor(nn.Module):
    def __init__(
        self, S_layers: NN_layers=None,
    ):  # regresor always returns one output
        super().__init__()
        
        self.layers = nn.ModuleDict()  # to write the different layers of the pytorch regression model
        self.hyperparams: dict = {}  # to write the hyperparams of the model
        
        """
        The structure in S_layers is a dict with three layers: input, hidden, and output, each one can have 
        six params: id, layer type, dropout, normalize, and activation, in that order.
        {
            'input_layer':{
                'id':'input',
                'layer': nn.Linear(val1, val2),
                'dropout': None,
                'normalize': None,
                'activation':nn.ReLU()
            },
            'hidden_layers':{
                'id':'hidden',
                'hlayers':[{
                    'id': 'hidden_1',
                    'layer': nn.Linear(val2, val3),
                    'dropout': nn.Dropout(0.2),
                    'normalize': nn.LayerNorm(val3),
                    'activation': nn.ReLU(),
                },
                {
                    'id': 'hidden_2',
                    ....            
            },
            ...
            ],},
            'output_layer':{
                'id':'output',
                'layer': nn.Linear(valk, 1),
                'dropout': None,
                'normalize': None,
                'activation':nn.ReLU()
            }
            
            'hyperparams':{
                'batch_size': int,
                'lr': float,
                'num_epochs': int,
                'loss_func': nn.MSELoss(),
                'optimizer': 'Adam',
    }        
        }
        """ 
        # Lets navigate over dict structure that shows the NN
        for key, layer in S_layers.items():  # there are three layers: input, hidden, and output
            if key == 'hidden_layers':
                for hidden in layer['hlayers']:
                    self.layers[f"{hidden['id']}"] = hidden['layer']
                    if 'dropout' in hidden.keys():
                        self.layers[f"{hidden['id']}_dropout"] = hidden['dropout']
                    if 'normalize' in hidden.keys():
                        self.layers[f"{hidden['id']}_normalize"] = hidden['normalize']
                    if 'activation' in hidden.keys():
                        self.layers[f"{hidden['id']}_activation"] = hidden['activation']
            
            elif key == 'hyperparams':              
                self.hyperparams['batch_size'] = layer['batch_size']
                self.hyperparams['lr'] = layer['lr']
                self.hyperparams['num_epochs'] = layer['num_epochs']
                self.hyperparams['loss_func'] = layer['loss_func']
                self.hyperparams['optimizer'] = Pytorch_Optimizers(layer['optimizer']).optimizer(
                    self.parameters(), lr=self.hyperparams['lr'])
                
            else:
                self.layers[f"{layer['id']}"] = layer['layer']
                if 'dropout' in layer.keys():
                    self.layers[f"{layer['id']}_dropout"] = layer['dropout']
                if 'normalize' in layer.keys():
                    self.layers[f"{layer['id']}_normalize"] = layer['normalize']
                if 'activation' in layer.keys():
                    self.layers[f"{layer['id']}_activation"] = layer['activation']
        di_f_logger.info(
        f"model definition: {self.layers} -- Hyperparams: {self.hyperparams}"
    )
                

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x


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

