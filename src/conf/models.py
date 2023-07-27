# here we define the different models
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from pycaret.internal.pipeline import Pipeline


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
        
        #for i in range(5):
        #    print('y:', y[i])
        return y