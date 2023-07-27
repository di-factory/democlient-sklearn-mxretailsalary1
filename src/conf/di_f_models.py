#version: july 15, 2023

from omegaconf import DictConfig
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import HuberRegressor, BayesianRidge, Ridge
from sklearn.model_selection import train_test_split

from pycaret.internal.pipeline import Pipeline
from pycaret.internal.preprocess.preprocessor import PowerTransformer, StandardScaler, SimpleImputer 
from pycaret.internal.preprocess.preprocessor import FixImbalancer, TransformerWrapper, TargetEncoder, OneHotEncoder, MinMaxScaler


import src.conf.preprocessors as pp
from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error



from catboost import CatBoostRegressor
import src.conf.ml_util as ml_util
import os
import pandas as pd
import numpy as np
import joblib

import mlflow
import mlflow.sklearn
from typing import List, Any
from pydantic import BaseModel, create_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset,Subset,SubsetRandomSampler



# from mlflow.models.signature import infer_signature

# --------- LEVEL 0 -----------------
class Di_FX:  # Main class for all the experiments definitions
    class Di_FX_score:  # To record the scores of each model
        def __init__(self, score: dict):
            self.id = score['id']
            self.metric = score['metric']

    class Di_FX_Kfold:  # To record the params of Kfolds
        def __init__(self, kfold_dict: dict = {'n_splits': 5, 'shuffle': True, 'random_state': 123}):
            self.n_splits = kfold_dict['n_splits']
            self.shuffle = kfold_dict['shuffle']
            self.random_state = kfold_dict['random_state']
            
    class Features(BaseModel):  # To represent the Features Set of the experiment
        pass
       
        def __repr__(self):
            return {'id': self.id, 'metric': self.metric}

    def __init__(self, cfg: DictConfig):
        #def create_Features() -> DictConfig: # this codes keeps when  pkl can manage create_model from pydantic
        #    fields_dict = {}
        #    for record in self.cfg.data_fields.features:
        #        fields_dict[record.field] = (eval(record.type), record.default)
        #    return create_model('Features', **fields_dict)  # This retrurn a class  
        
        def create_catalogues() -> dict:
            catalogue = {}
            for record in self.cfg.data_fields.features:
                if hasattr(record, 'aList'):  # Looking for the field aList in each record
                    catalogue[record.field] = record.aList 
            return catalogue
        
        self.di_fx: List[str] = []
        self.di_fx.append('Di_FX')  # Level 0 class of the experiment 
        
        self.cfg: DictConfig = cfg  # config.yaml file 
        self.id: str = self.cfg.general_ml.experiment  # id = client.project.experiment
        self.features: self.Features  # create_Features()()  # Features as instance of BaseModel by create_features
        self.feature_list = [r.field for r in self.cfg.data_fields.features]
        
        self.dataPipeline: Pipeline = None  # Data Pipeline
        self.model: Any = None  # ML Pipeline
        
        self.catalogues: dict = create_catalogues()
        self.scores: List[self.Di_FX_score] = None
        self.kfold = self.Di_FX_Kfold()
  
    def load_dataPipeline(self) -> None:  # this method loads the dataPipeline model that was created/saved in runDataPipeline()
        try:
            self.dataPipeline = joblib.load(os.path.join(self.cfg.paths.models_dir, self.cfg.file_names.datapipeline))
            print("datapipeline loaded successfully!")
        except Exception as e:
            print(f"Error loading the datapipeline: {e}")
        
    def save_dataPipeline(self) -> None:  # this method saves the dataPipeline model
        try:
            joblib.dump(self.dataPipeline, os.path.join(self.cfg.paths.models_dir, self.cfg.file_names.datapipeline))
            print("Datapipeline saved successfully!")
        except Exception as e:
            print(f"Error saving the datapipeline: {e}")
        
    def load_model(self) -> None:  # this method loads the model prediction that was created/saved in fits methods
        try:
            self.model = joblib.load(os.path.join(self.cfg.paths.models_dir, self.cfg.file_names.model))
            print("model loaded successfully!")
        except Exception as e:
            print(f"Error loading the model: {e}")
    
    def save_model(self) -> None:  # this method saves the model prediction
        try:
            joblib.dump(self.model, os.path.join(self.cfg.paths.models_dir, self.cfg.file_names.model))
            print("model saved successfully!")
        except Exception as e:
            print(f"Error saving the model: {e}")
        
    def runDataPipeline(self) -> None:  # this method runs the dataPipeline object-class
        pass

    def fit(self, tracking: bool = False) -> dict:  # this methods train the model prediction defined.
        pass

    def fit_Kfold(self) -> dict:  # this method use Crossvalidations in trainning
        pass

    def predict(self, X: pd.DataFrame) -> np.array:  # this method makes predictions of unseen incomig data 
        pass


# --------- LEVEL 1 -----------------
class Di_FX_ml(Di_FX):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.di_fx.append('ml')  # Level 1 class of the experiment 

    def runDataPipeline(self) -> None:  # this method runs the dataPipeline object-class

        def transform_data(data: pd.DataFrame) -> pd.DataFrame:
            """
            to transform data using a sklearn pipeline technology 
            data is the whole dataset, fetures + label
            """
            print(' STEP2: Transforming data using a PIPELINE')
   
            dpl = self.dataPipeline  # instanciating dataPipeline object
            labels = data[self.cfg.data_fields.label]
            dpl.fit(data[self.feature_list], labels)  # only fitting features, no label
            df = dpl.transform(data[self.feature_list])  # transformations is making on features
            #print(df.describe())

            print(f'       After transformation: (rows,cols) {df.shape}')
            df[self.cfg.data_fields.label] = labels  # adding column labels to dataframe
            return df

        # loading pre-processed data from notebook data profiling
        pathfile_l = os.path.join(self.cfg.paths.processed_data_dir, self.cfg.file_names.processed_data)
        data = ml_util.load_data(pathfile_l, 
                                 # encoding=<particular encoding>
                                 )

        # setting the directory where to save transformed data
        pathfile_s = os.path.join(self.cfg.paths.interim_data_dir, self.cfg.file_names.data_file)

        # executing transformation
        data = transform_data(data=data)

        # write the whole transformed data
        ml_util.write_transformed(pathfile_s, data)

        # setting paths where to save splitted data
        pathfile_train_futures = os.path.join(self.cfg.paths.processed_data_dir, self.cfg.file_names.train_features)
        pathfile_train_labels = os.path.join(self.cfg.paths.processed_data_dir, self.cfg.file_names.train_labels)
        pathfile_validation_futures = os.path.join(self.cfg.paths.processed_data_dir, self.cfg.file_names.validation_features)
        pathfile_validation_labels = os.path.join(self.cfg.paths.processed_data_dir, self.cfg.file_names.validation_labels)
        pathfile_test_futures = os.path.join(self.cfg.paths.processed_data_dir, self.cfg.file_names.test_features)
        pathfile_test_labels = os.path.join(self.cfg.paths.processed_data_dir, self.cfg.file_names.test_labels)
    
        # executting and writting spplited data
        ml_util.write_spplited(
            pathfile_train_futures, pathfile_train_labels, pathfile_validation_futures, 
            pathfile_validation_labels, pathfile_test_futures,
            pathfile_test_labels, data, self.cfg.data_fields.label, 
                self.cfg.data_pipeline.data_transform_params.percent_valid,
                self.cfg.data_pipeline.data_transform_params.percent_test, 
                self.cfg.general_ml.seed)    
        
        # save data pipeline model
        self.save_dataPipeline()

    def fit(self, tracking: bool = False) -> dict:  # this methods train the model prediction defined.
        pass
    
    def fit_Kfold(self, tracking: bool = False) -> dict:  # this method use Crossvalidations in trainning
        pass
    
    def predict(self, X: pd.DataFrame) -> np.array:  # this method makes predictions of unseen incomig data 
        #print(X.head(), X.dtypes)
        self.load_dataPipeline()
        self.load_model()
        
        X_transformed = self.dataPipeline.transform(X)
        #print(X_transformed)
        result = self.model.predict(X_transformed)
        print(result)
        return np.array(result)


class Di_FX_Pt_NN(Di_FX):  # Pytorch Neural Networks
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.di_fx.append('Pytorch_NN')  # Level 1 class of the experiment 

    def runDataPipeline(self) -> None:  # this method runs the dataPipeline object-class

        def transform_data(data: pd.DataFrame) -> pd.DataFrame:
            """
            to transform data using a sklearn pipeline technology 
            data is the whole dataset, fetures + label
            """
            print(' STEP2: Transforming data using a PIPELINE')
   
            dpl = self.dataPipeline  # instanciating dataPipeline object
            labels = data[self.cfg.data_fields.label]
            dpl.fit(data[self.feature_list], labels)  # only fitting features, no label
            df = dpl.transform(data[self.feature_list])  # transformations is making on features
            #print(df.describe())

            print(f'       After transformation: (rows,cols) {df.shape}')
            df[self.cfg.data_fields.label] = labels  # adding column labels to dataframe
            return df

        # loading pre-processed data from notebook data profiling
        pathfile_l = os.path.join(self.cfg.paths.processed_data_dir, self.cfg.file_names.processed_data)
        data = ml_util.load_data(pathfile_l, 
                                 # encoding=<particular encoding>
                                 )

        # setting the directory where to save transformed data
        pathfile_s = os.path.join(self.cfg.paths.interim_data_dir, self.cfg.file_names.data_file)

        # executing transformation
        data = transform_data(data=data)

        # write the whole transformed data
        ml_util.write_transformed(pathfile_s, data)

        # setting paths where to save splitted data
        pathfile_train_futures = os.path.join(self.cfg.paths.processed_data_dir, self.cfg.file_names.train_features)
        pathfile_train_labels = os.path.join(self.cfg.paths.processed_data_dir, self.cfg.file_names.train_labels)
        pathfile_validation_futures = os.path.join(self.cfg.paths.processed_data_dir, self.cfg.file_names.validation_features)
        pathfile_validation_labels = os.path.join(self.cfg.paths.processed_data_dir, self.cfg.file_names.validation_labels)
        pathfile_test_futures = os.path.join(self.cfg.paths.processed_data_dir, self.cfg.file_names.test_features)
        pathfile_test_labels = os.path.join(self.cfg.paths.processed_data_dir, self.cfg.file_names.test_labels)
    
        # executting and writting spplited data
        ml_util.write_spplited(
            pathfile_train_futures, pathfile_train_labels, pathfile_validation_futures, 
            pathfile_validation_labels, pathfile_test_futures,
            pathfile_test_labels, data, self.cfg.data_fields.label, 
                self.cfg.data_pipeline.data_transform_params.percent_valid,
                self.cfg.data_pipeline.data_transform_params.percent_test, 
                self.cfg.general_ml.seed)    
        
        # save data pipeline model
        self.save_dataPipeline()

    def fit(self, tracking: bool = False) -> dict:  # this methods train the model prediction defined.
        pass
    
    def fit_Kfold(self, tracking: bool = False) -> dict:  # this method use Crossvalidations in trainning
        pass
    
    def predict(self, X: pd.DataFrame) -> np.array:  # this method makes predictions of unseen incomig data 
        #print(X.head(), X.dtypes)
        self.load_dataPipeline()
        self.load_model()
        
        X_transformed = self.dataPipeline.transform(X)
        #print(X_transformed)
        result = self.model.predict(X_transformed)
        print(result)


# --------- LEVEL 2 -----------------

# --------- LEVEL 2-a Neural Networks with Pytorch  -----------------
class prueba_NN(Di_FX_Pt_NN, nn.Module): 
    def __init__(self, cfg: DictConfig, input_dim: int):
        super().__init__(cfg)
        nn.Module.__init__(self)
        
        self.cfg: DictConfig = cfg  # config.yaml file 
        self.id: str = self.cfg.general_ml.experiment  # id = client.project.experiment
               
        self.layeri = nn.Linear(input_dim, 100)
        self.layerh1 = nn.Linear(100, 250)
        self.layerh2 = nn.Linear(250, 100)
        self.layero = nn.Linear(100, 1)

        self.batch_size = 50
        self.lr = 0.001
        self.num_epochs = 3000
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)
    
    def generate_data(self, num_samples, noise_std=1.0):
        
        X = torch.linspace(-50, 50, num_samples).view(-1, 1)
        y_true = X**4 - 5*X**3 + 3*X**2 - X + 2
        noise = torch.randn_like(y_true) * noise_std
        y_noisy = y_true + noise

        return normalize(X), normalize(y_noisy)

    def forward(self, x):
        y = F.relu(self.layeri(x))
        y = F.relu(self.layerh1(y))
        y = F.relu(self.layerh2(y))
        y = self.layero(y)
        
        #for i in range(5):
        #    print('y:', y[i])
        return y
    
    def evaluate(self):
        
        test_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                 self.cfg.file_names.test_features))
        test_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                               self.cfg.file_names.test_labels))
        return None

    def fit(self, tracking:bool = False):
        # Load the data
        train_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir,
                                                  self.cfg.file_names.train_features))
        train_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                self.cfg.file_names.train_labels))
        validation_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                       self.cfg.file_names.validation_features))
        validation_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                     self.cfg.file_names.validation_labels))
        test_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                 self.cfg.file_names.test_features))
        test_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                               self.cfg.file_names.test_labels))
        
        

        #changing from datasets to tensors
        train_features = torch.tensor(train_features.values, dtype=torch.float)
        train_labels = torch.log(torch.tensor(train_labels.values, dtype=torch.float))  # scaling to log the labels
        validation_features = torch.tensor(validation_features.values, dtype=torch.float)
        validation_labels = torch.log(torch.tensor(validation_labels.values, dtype=torch.float))  # scaling to log the labels
        test_features = torch.tensor(test_features.values, dtype=torch.float)
        test_labels = torch.log(torch.tensor(test_labels.values, dtype=torch.float))  # scaling to log the labels
    
       
       #creating datasets and dataloaders
        train_data = TensorDataset(train_features, train_labels)
        validation_data = TensorDataset(validation_features, validation_labels)
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        validation_loader = DataLoader(validation_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
 
        if tracking:
            #setting up mlflow    
            mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
            mlflow.set_experiment(f'{self.cfg.mlflow.tracking_experiment_name}_{self.di_f_exp}')
            mlflow.sklearn.autolog()
            mlflow.start_run()

        # Fit the model
        print('fitting the  model')
        print(f'dimensions: features-> {train_features.shape}, labels-> {train_labels.shape}')
   
        self.losses = torch.zeros(self.num_epochs)  # To grab the loss_train for each epoch an see evolution
        self.r2 = torch.zeros(self.num_epochs)  # To grab the r2 for each epoch an see evolution
        for e in range(self.num_epochs):
            # switching to train mode
            self.train()
            batch_loss = []  # To grab the loss_train for each batch and then get mean() 
            for X, y in train_loader:  # steping by each pair X,y of size batch_size
                y_pred = self.forward(X)
                
                loss = self.loss_func(y_pred, y)
                
                # backpropagation block
                self.optimizer.zero_grad() # reinit gradients
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            
            self.losses[e] = np.mean(batch_loss)
            
            # let's evaluate:
            self.eval()
            X,y = next(iter(validation_loader))
            with torch.no_grad(): # stop gradient descent in eval)
                y_pred = self.forward(X)
                
            self.r2[e] = r2_score(y, y_pred)
            if (e+1)%100 == 0:
                print(f"epoch:{e+1}, loss:{self.losses[e]}, r2:{self.r2[e]}")
       
        #calculating trainning score
        y_pred = self.forward(validation_features)
        #mse_test = mean_squared_error(y_pred.detach().numpy(), validation_labels.detach().numpy())
        mse_val = F.mse_loss(validation_labels, y_pred)
        r2_val = r2_score(y_pred.detach().numpy(), validation_labels.detach().numpy())
        print(mse_val, r2_val) 
    
    
        #calculating testing score
        y_pred = self.forward(test_features)
        #mse_test = mean_squared_error(y_pred.detach().numpy(), test_labels.detach().numpy())
        
        mse_test = F.mse_loss(test_labels, y_pred)
        r2_test = r2_score(y_pred.detach().numpy(), test_labels.detach().numpy())
        print(mse_test, r2_test) 
    
        """
        #calculating trainning score
    
        y_pred = self.model.predict(X_train)
        print('scores for trainning:')
        for score in self.scores:
            sc = score['metric'](y_train, y_pred)
            print(f"train scoring {score['id']}: {sc}")
        

        #calculating testing score
        y_pred = self.model.predict(test_features)
        print('scores for test:')
        for score in self.scores:
            sc = score['metric'](test_labels, y_pred)
            print(f"test scoring {score['id']}: {sc}")
        
        # saving the model
        self.save_model()
        
        if tracking:
            # stoping mlflow run experiment
            mlflow.end_run()
        
        return None"""
    
   
class Di_FX_FFNN(Di_FX_Pt_NN):  # Pytorch Feedforward NN in Pytorch
    def __init__(self, cfg: DictConfig):
 
        class regressionNet(nn.Module):
            def __init__(self, nImputs, nOutputs, hidden, nHidden, dr=0.0):
                super().__init__()

                #Create a dictionary to save the layers
                self.layers = nn.ModuleDict()
                self.nImputs = nImputs
                self.nOutputs = nOutputs
                self.nHidden = nHidden
                self.hidden = hidden
                self.dr = dr #Dropuot rate

                self.trainAcc = [] #for storing the accuracy vectors of trainning and dev
                self.devAcc = []
                self.losses = [] #for storing losses at trainning

                #This two vectors are for measure weight changes and withconds of the learnig process
                self.weightChanges = [] #This counts about how much is learning in each epoch
                self.weightConds = [] #this how much is specializing in anykind of particularlity
                
                ### input layer
                self.layers["input"] = nn.Linear(nImputs,nHidden)

                for h in range(self.hidden):      
                ### hidden layer
                    self.layers[f'hidden{h}'] = nn.Linear(nHidden,nHidden)

                ### output layer
                self.layers['output'] = nn.Linear(nHidden,nOutputs)

            def nCopy(self, n):
                self.load_state_dict(n['net'])

                # forward pass

            def forward(self, x):
                x = F.leaky_relu( self.layers['input'](x) )
                x = F.dropout(x,p=self.dr,training=self.training) # switch dropout off during .eval()
            
                for h in range(self.hidden):
                    x = F.leaky_relu( self.layers[f'hidden{h}'](x) )
                    x = F.dropout(x,p=self.dr,training=self.training) # switch dropout off during .eval()
                return self.layers['output'](x)
        
            def testModel(self, n, test_loader, precision=0.1):
                # extract X,y from test dataloader
                X,y = next(iter(test_loader)) 
                self.load_state_dict(n['net'])
                self.eval()
                yHat = self.forward(X)
            
                bestAcc =  100*torch.mean((torch.abs((yHat-y)**2)<precision).float())
                return(yHat, bestAcc)

            def fit(self, train_loader, dev_loader, numepochs = 100, learningRate=0.01, weight_decay=0.1, precision=0.1, initializer='kaiming'):
                # New! initialize a dictionary for the best model
                
                self.weightChanges = np.zeros((numepochs, self.hidden+2)) #This two vectors are for measure weight changes and withconds of the learnig process
                self.weightConds = np.zeros((numepochs, self.hidden+2))
                
                theBestModel = {'devAccuracy':0, 'net':None, 'epoch':0} #net will be the whole model instance

                if initializer == 'xavier':         #this set the methgod for iniatiazing weights
                    for p in self.named_parameters():
                        if 'weight' in p[0]:
                            nn.init.xavier_normal_(p[1].data)
                else:
                    for p in self.named_parameters():
                        if 'weight' in p[0]:
                            nn.init.kaiming_uniform_(p[1].data, nonlinearity='leaky_relu') #be sure that relu or leaky_relu is the activation

                # loss function
                self.lossfun = nn.MSELoss()

                # optimizer
                self.optimizer = torch.optim.AdamW(self.parameters(), lr=learningRate, weight_decay=weight_decay, )


                # initialize losses
                self.losses   = torch.zeros(numepochs)

                # loop over epochs
                for epochi in range(numepochs):

                    # store the weights for each layer
                    preW = []
                    for p in self.named_parameters():
                        if 'weight' in p[0]:
                            preW.append( copy.deepcopy(p[1].data.numpy()) )


                    # switch on training mode
                    self.train()

                    # loop over training data batches
                    batchAcc  = []
                    batchLoss = []

                    for X,y in train_loader: #each time that runs in EACH epoch, suffle ALL the samples in each bactch
                
                    # forward pass and loss
                        yHat = self.forward(X)
                    
                        loss = self.lossfun(yHat,y) #Use in case of BCEWithLogitsLoss()

                        # backprop
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        # loss from this batch
                        batchLoss.append(loss.item())

                        # compute accuracy
                        matches = (torch.abs((yHat-y)**2)<precision) # booleans (false/true)
                        matchesNumeric = matches.float()             # convert to numbers (0/1)
                        accuracyPct = 100*torch.mean(matchesNumeric) # average and x100 
                        batchAcc.append( accuracyPct )               # add to list of accuracies
                        # end of batch loop...

                    # now that we've trained through the batches, get their average training accuracy
                    self.trainAcc.append( np.mean(batchAcc) )

                    # and get average losses across the batches
                    self.losses[epochi] = np.mean(batchLoss)

                    # test accuracy
                    self.eval()
                    X,y = next(iter(dev_loader)) # extract X,y from devset dataloader

                    with torch.no_grad(): # deactivates autograd
                        yHat = self.forward(X)

                    # compare the following really long line of code to the training accuracy lines

                    self.devAcc.append( 100*torch.mean((torch.abs((yHat- y)**2)<precision).float()))


                    # New! Store this model if it's the best so far
                    if self.devAcc[-1]>theBestModel['devAccuracy']:
                
                        # new best accuracy
                        theBestModel['devAccuracy'] = self.devAcc[-1].item()
                
                        # epoch iteration
                        theBestModel['epoch'] = epochi

                        # model's internal state
                        theBestModel['net'] = copy.deepcopy( self.state_dict() ) #here update the best model

                    # Get the post-learning state of the weights
                    for (i,p) in enumerate(self.named_parameters()):
                        if 'weight' in p[0]:
                            # condition number, just for weights not bias
                            self.weightConds[epochi,int(i/2)]  = np.linalg.cond(p[1].data)

                            # Frobenius norm of the weight change from pre-learning
                            self.weightChanges[epochi,int(i/2)] = np.linalg.norm( preW[int(i/2)]-p[1].data.numpy(), ord='fro')

                #end epoch

                # function output
                return self.trainAcc,self.devAcc,self.losses,theBestModel
            
            def drawModel(self, figsize=(15,18)):
                #plot results of trainning
                # set up the plot
                fig,ax = plt.subplots(3,2,figsize=figsize)
                ax = ax.flatten()

                # accuracy
                ax[0].plot(self.losses.detach(),'-')
                ax[0].set_ylabel('Loss')
                ax[0].set_xlabel('epoch')
                ax[0].set_title('Losses')
                ax[0].set_ylim([0,0.6])
                
                ax[1].plot(self.trainAcc,'-',label='Train')
                ax[1].plot(self.devAcc,'-',label='Devset')
                ax[1].set_ylabel('devAccuracy (%)')
                ax[1].set_xlabel('Epoch')
                ax[1].set_title('Accuracy')
                #ax[1].set_ylim([85,95])
                #ax[1].set_xlim([80,105])
                ax[1].legend()
                
                layername = []
                for (i,p) in enumerate(self.named_parameters()):
                    if 'weight' in p[0]:
                        layername.append(p[0][:-7])
                
                # weight changes
                # get a list of layer names
                
                ax[2].plot(self.weightChanges)
                ax[2].set_xlabel('Epochs')
                ax[2].set_title('Weight change from previous epoch')
                ax[2].legend(layername)
                #ax[2].set_ylim([0,0.2])
                #ax[2].set_yscale('log')

                # weight condition numbers
                ax[3].plot(self.weightConds)
                ax[3].set_xlabel('Epochs')
                ax[3].set_title('Condition number')
                ax[3].legend(layername)
                #ax[3].set_ylim([0,1000])
                #ax[3].set_yscale('log')

                #Graph an histogram of weights
                # store the weights for each layer
                preW = []
                for p in self.named_parameters():
                    if 'weight' in p[0]:
                        preW = np.append(preW, p[1].data.numpy())

                y,x = np.histogram(preW,30)
                ax[4].set_title('Total weight distribution after trainning')
                ax[4].plot((x[1:]+x[:-1])/2,y,'b')
                ax[4].set_xlabel('Weight value')
                ax[4].set_ylabel('Count')

                
                from scipy.stats import zscore # normalize for scaling offsets
                #using zcore is only for vision convinience

                ax[5].plot(zscore(np.diff(self.trainAcc)),label='d(trainAcc)')
                ax[5].plot(zscore(np.mean(self.weightChanges,axis=1)),label='Weight change')
                ax[5].legend()
                ax[5].set_title('Change in weights by change in accuracy')
                ax[5].set_xlabel('Epoch')
                ax[5].set_ylim([-3,3])
                

                plt.tight_layout()
                plt.show()
              
        super().__init__(cfg)
        self.di_fx.append('Feed Forward NN')  # Level 2 class of the experiment 
        self.model: regressionNet = regressionNet(nImputs, nOutputs. hidden, nHidden, dr)

    def fit(self, tracking: bool = False) -> dict:  # this methods train the model prediction defined.
        # Load the data
        train_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir,
                                                  self.cfg.file_names.train_features))
        train_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                self.cfg.file_names.train_labels))
        validation_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                       self.cfg.file_names.validation_features))
        validation_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                     self.cfg.file_names.validation_labels))
        test_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                 self.cfg.file_names.test_features))
        test_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                               self.cfg.file_names.test_labels))

        #creating datasets from concatenation of sources
        #In this case as kfold, we need to concatenate the whole datasets
        X_train = pd.concat([train_features, validation_features], ignore_index=True)
        y_train = pd.concat([train_labels, validation_labels], ignore_index=True)

        if tracking:
            #setting up mlflow    
            mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
            mlflow.set_experiment(f'{self.cfg.mlflow.tracking_experiment_name}_{self.di_f_exp}')
            mlflow.sklearn.autolog()
            mlflow.start_run()

        # Fit the model
        print('fitting the  model')
        print(f'dimensions: features-> {X_train.shape}, labels-> {y_train.shape}')
   
        self.model.fit(X_train, np.ravel(y_train))
    
        #calculating trainning score
        y_pred = self.model.predict(X_train)
        print('scores for trainning:')
        for score in self.scores:
            sc = score['metric'](y_train, y_pred)
            print(f"train scoring {score['id']}: {sc}")
        

        #calculating testing score
        y_pred = self.model.predict(test_features)
        print('scores for test:')
        for score in self.scores:
            sc = score['metric'](test_labels, y_pred)
            print(f"test scoring {score['id']}: {sc}")
        
        # saving the model
        self.save_model()
        
        if tracking:
            # stoping mlflow run experiment
            mlflow.end_run()
        
        return {'id': score['id'], 'result': sc}

    def fit_Kfold(self, tracking: bool = False) -> dict:  # this method use Crossvalidations in trainning
        # Load the data
        train_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir,
                                                  self.cfg.file_names.train_features))
        train_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                self.cfg.file_names.train_labels))
        validation_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                       self.cfg.file_names.validation_features))
        validation_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                     self.cfg.file_names.validation_labels))
        test_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                 self.cfg.file_names.test_features))
        test_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                               self.cfg.file_names.test_labels))
        
        #creating the kfold
        kfold = KFold(n_splits = kfold_dict['n_splits'], 
                      shuffle = kfold_dict['shuffle'], 
                      random_state=self.cfg.general_ml.seed
                      )

        #creating datasets from concatenation of sources
        #In this case as kfold, we need to concatenate the whole datasets
        whole_features = pd.concat([train_features, validation_features, test_features], ignore_index=True)
        whole_labels = pd.concat([train_labels, validation_labels, test_labels], ignore_index=True)
                       
        if tracking:
            #setting up mlflow    
            mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
            mlflow.set_experiment(f'{self.cfg.mlflow.tracking_experiment_name}_{self.di_f_exp}')
            mlflow.sklearn.autolog()
            mlflow.start_run()
                     
        #printing r2 sccore with cv=5
        print('scores pre-Kfold:')
        for score in self.scores:
            sc = cross_val_score(self.model, whole_features, np.ravel(whole_labels), cv=5, scoring=score['id']).mean()
            print(f"cross_val_scoring (before kfold){score['id']}: {sc}")

        # Fit the model
        print('fitting the  model')
        print(f'dimensions: features-> {whole_features.shape}, labels-> {whole_labels.shape}')
   
        scores_v = {}
    
        for train_index, test_index in kfold.split(whole_features):
           
            X_train= np.array(whole_features)[train_index]
            X_test = np.array(whole_features)[test_index]
            y_train = np.array(whole_labels)[train_index]
            y_test = np.array(whole_labels)[test_index]

            self.model.fit(X_train, np.ravel(y_train))
            y_pred = self.model.predict(X_test)

            for score in self.scores:
                sc = score['metric'](y_test, y_pred)
                scores_v[score['id']]+=sc
   
        print('scores post-Kfold:')
        for score in self.scores:
            print(f"scoring after kfold {score['id']}: {scores_v[score['id']]/5}")
        
        # saving the model
        self.save_model()
        
        if tracking:
            # stoping mlflow run experiment
            mlflow.end_run()
        
        return {'id': score['id'], 'result': np.mean(scores)}

    def predict(self, X: pd.DataFrame) -> np.array:  # this method makes predictions of unseen incomig data 
        super().predict(X)


# --------- LEVEL 2-a ML with scikit-learn  -----------------
class Di_FX_Voting(Di_FX_ml):  
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.di_fx.append('Voting')  # Level 2 class of the experiment 

    def fit(self, tracking: bool = False) -> dict:  # this methods train the model prediction defined.
        # Load the data
        train_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir,
                                                  self.cfg.file_names.train_features))
        train_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                self.cfg.file_names.train_labels))
        validation_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                       self.cfg.file_names.validation_features))
        validation_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                     self.cfg.file_names.validation_labels))
        test_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                 self.cfg.file_names.test_features))
        test_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                               self.cfg.file_names.test_labels))

        #creating datasets from concatenation of sources
        #In this case as kfold, we need to concatenate the whole datasets
        X_train = pd.concat([train_features, validation_features], ignore_index=True)
        y_train = pd.concat([train_labels, validation_labels], ignore_index=True)

        if tracking:
            #setting up mlflow    
            mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
            mlflow.set_experiment(f'{self.cfg.mlflow.tracking_experiment_name}_{self.di_f_exp}')
            mlflow.sklearn.autolog()
            mlflow.start_run()

        # Fit the model
        print('fitting the  model')
        print(f'dimensions: features-> {X_train.shape}, labels-> {y_train.shape}')
   
        self.model.fit(X_train, np.ravel(y_train))
    
        #calculating trainning score
        y_pred = self.model.predict(X_train)
        print('scores for trainning:')
        for score in self.scores:
            sc = score['metric'](y_train, y_pred)
            print(f"train scoring {score['id']}: {sc}")
        

        #calculating testing score
        y_pred = self.model.predict(test_features)
        print('scores for test:')
        for score in self.scores:
            sc = score['metric'](test_labels, y_pred)
            print(f"test scoring {score['id']}: {sc}")
        
        # saving the model
        self.save_model()
        
        if tracking:
            # stoping mlflow run experiment
            mlflow.end_run()
        
        return {'id': score['id'], 'result': sc}

    def fit_Kfold(self, tracking: bool = False) -> dict:  # this method use Crossvalidations in trainning
        # Load the data
        train_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir,
                                                  self.cfg.file_names.train_features))
        train_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                self.cfg.file_names.train_labels))
        validation_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                       self.cfg.file_names.validation_features))
        validation_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                     self.cfg.file_names.validation_labels))
        test_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                 self.cfg.file_names.test_features))
        test_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                               self.cfg.file_names.test_labels))
        
        #creating the kfold
        kfold = KFold(n_splits = kfold_dict['n_splits'], 
                      shuffle = kfold_dict['shuffle'], 
                      random_state=self.cfg.general_ml.seed
                      )

        #creating datasets from concatenation of sources
        #In this case as kfold, we need to concatenate the whole datasets
        whole_features = pd.concat([train_features, validation_features, test_features], ignore_index=True)
        whole_labels = pd.concat([train_labels, validation_labels, test_labels], ignore_index=True)
                       
        if tracking:
            #setting up mlflow    
            mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
            mlflow.set_experiment(f'{self.cfg.mlflow.tracking_experiment_name}_{self.di_f_exp}')
            mlflow.sklearn.autolog()
            mlflow.start_run()
                     
        #printing r2 sccore with cv=5
        print('scores pre-Kfold:')
        for score in self.scores:
            sc = cross_val_score(self.model, whole_features, np.ravel(whole_labels), cv=5, scoring=score['id']).mean()
            print(f"cross_val_scoring (before kfold){score['id']}: {sc}")

        # Fit the model
        print('fitting the  model')
        print(f'dimensions: features-> {whole_features.shape}, labels-> {whole_labels.shape}')
   
        scores_v = {}
    
        for train_index, test_index in kfold.split(whole_features):
           
            X_train= np.array(whole_features)[train_index]
            X_test = np.array(whole_features)[test_index]
            y_train = np.array(whole_labels)[train_index]
            y_test = np.array(whole_labels)[test_index]

            self.model.fit(X_train, np.ravel(y_train))
            y_pred = self.model.predict(X_test)

            for score in self.scores:
                sc = score['metric'](y_test, y_pred)
                scores_v[score['id']]+=sc
   
        print('scores post-Kfold:')
        for score in self.scores:
            print(f"scoring after kfold {score['id']}: {scores_v[score['id']]/5}")
        
        # saving the model
        self.save_model()
        
        if tracking:
            # stoping mlflow run experiment
            mlflow.end_run()
        
        return {'id': score['id'], 'result': np.mean(scores)}

    def predict(self, X: pd.DataFrame) -> np.array:  # this method makes predictions of unseen incomig data 
        super().predict(X)


class Di_FX_Regressor(Di_FX_ml):  
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.di_fx.append('Regressor')  # Level 2 class of the experiment 

    def fit(self, tracking: bool = False) -> dict:  # this methods train the model prediction defined.
        # Load the data
        train_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir,
                                                  self.cfg.file_names.train_features))
        train_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                self.cfg.file_names.train_labels))
        validation_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                       self.cfg.file_names.validation_features))
        validation_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                     self.cfg.file_names.validation_labels))
        test_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                 self.cfg.file_names.test_features))
        test_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                               self.cfg.file_names.test_labels))

        #creating datasets from concatenation of sources
        #In this case as kfold, we need to concatenate the whole datasets
        X_train = pd.concat([train_features, validation_features], ignore_index=True)
        y_train = pd.concat([train_labels, validation_labels], ignore_index=True)

        if tracking:
            #setting up mlflow    
            mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
            mlflow.set_experiment(f'{self.cfg.mlflow.tracking_experiment_name}_{self.di_f_exp}')
            mlflow.sklearn.autolog()
            mlflow.start_run()

        # Fit the model
        print('fitting the  model')
        print(f'dimensions: features-> {X_train.shape}, labels-> {y_train.shape}')
   
        self.model.fit(X_train, np.ravel(y_train))
    
        #calculating trainning score
        y_pred = self.model.predict(X_train)
        print('scores for trainning:')
        for score in self.scores:
            sc = score['metric'](y_train, y_pred)
            print(f"train scoring {score['id']}: {sc}")
        

        #calculating testing score
        y_pred = self.model.predict(test_features)
        print('scores for test:')
        for score in self.scores:
            sc = score['metric'](test_labels, y_pred)
            print(f"test scoring {score['id']}: {sc}")
        
        # saving the model
        self.save_model()
        
        if tracking:
            # stoping mlflow run experiment
            mlflow.end_run()
        
        return {'id': score['id'], 'result': sc}

    def fit_Kfold(self, tracking: bool = False) -> dict:  # this method use Crossvalidations in trainning
        # Load the data
        train_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir,
                                                  self.cfg.file_names.train_features))
        train_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                self.cfg.file_names.train_labels))
        validation_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                       self.cfg.file_names.validation_features))
        validation_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                     self.cfg.file_names.validation_labels))
        test_features = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                                 self.cfg.file_names.test_features))
        test_labels = pd.read_csv(os.path.join(self.cfg.paths.processed_data_dir, 
                                               self.cfg.file_names.test_labels))
        
        #creating the kfold
        kfold = KFold(n_splits = kfold_dict['n_splits'], 
                      shuffle = kfold_dict['shuffle'], 
                      random_state=self.cfg.general_ml.seed
                      )

        #creating datasets from concatenation of sources
        #In this case as kfold, we need to concatenate the whole datasets
        whole_features = pd.concat([train_features, validation_features, test_features], ignore_index=True)
        whole_labels = pd.concat([train_labels, validation_labels, test_labels], ignore_index=True)
                       
        if tracking:
            #setting up mlflow    
            mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
            mlflow.set_experiment(f'{self.cfg.mlflow.tracking_experiment_name}_{self.di_f_exp}')
            mlflow.sklearn.autolog()
            mlflow.start_run()
                     
        #printing r2 sccore with cv=5
        print('scores pre-Kfold:')
        for score in self.scores:
            sc = cross_val_score(self.model, whole_features, np.ravel(whole_labels), cv=5, scoring=score['id']).mean()
            print(f"cross_val_scoring (before kfold){score['id']}: {sc}")

        # Fit the model
        print('fitting the  model')
        print(f'dimensions: features-> {whole_features.shape}, labels-> {whole_labels.shape}')
   
        scores_v = {}
    
        for train_index, test_index in kfold.split(whole_features):
           
            X_train= np.array(whole_features)[train_index]
            X_test = np.array(whole_features)[test_index]
            y_train = np.array(whole_labels)[train_index]
            y_test = np.array(whole_labels)[test_index]

            self.model.fit(X_train, np.ravel(y_train))
            y_pred = self.model.predict(X_test)

            for score in self.scores:
                sc = score['metric'](y_test, y_pred)
                scores_v[score['id']]+=sc
   
        print('scores post-Kfold:')
        for score in self.scores:
            print(f"scoring after kfold {score['id']}: {scores_v[score['id']]/5}")
        
        # saving the model
        self.save_model()
        
        if tracking:
            # stoping mlflow run experiment
            mlflow.end_run()
        
        return {'id': score['id'], 'result': np.mean(scores)}

    def predict(self, X: pd.DataFrame) -> np.array:  # this method makes predictions of unseen incomig data 
        super().predict(X)


class Metaclass(Di_FX):  # for using templates
    def __init__(self, id, cfg):
        super().__init__(id, cfg)
        self.dataPipeline = Pipeline(
            steps = [
                    #('keep_fields', 
                        #pp.KeepNecessaryFeatures(
                        #variables_to_keep = cfg.data_pipeline.keep_features
                        #)),
                 
                    ('numerical_imputer',
                        TransformerWrapper(
                            #exclude = ['match'],
                            include = ['income_employee_day',
                                        'employees_business'],
                            transformer = SimpleImputer(
                                #add_indicator = False,
                                #copy = True,
                                #fill_value = None,
                                #keep_empty_features = False,
                                #missing_values = np.nan,
                                #strategy = 'mean',
                                #verbose = 'deprecated',
                                ))),

                    ('categorical_imputer',
                        TransformerWrapper(
                            #exclude = ['match'], 
                            include = ['state'],
                            transformer = SimpleImputer(
                                #add_indicator = False, 
                                #copy = True,
                                #fill_value = None,
                                #keep_empty_features = False,
                                #missing_values=np.nan,
                                strategy = 'most_frequent',
                                #verbose = 'deprecated',
                                ))),

                    ('onehot_encoding',
                        TransformerWrapper(
                            include = ['state'],
                            transformer = OneHotEncoder(
                                cols = ['state'],
                                handle_missing = 'return_nan',
                                use_cat_names = True
                                ))),

                    #('rest_ecoding',
                        #TransformerWrapper(
                            #exclude = ['match'], 
                            #include = ['from', 'zipcode', 'undergra'],
                            #transformer = TargetEncoder(
                                #cols = ['from', 'zipcode','undergra'],
                                #drop_invariant = False,
                                #handle_missing = 'return_nan',
                                #handle_unknown ='value',
                                #hierarchy = None,
                                #min_samples_leaf = 20,
                                #return_df = True, smoothing =1 0,
                                #verbose = True
                                #))),

                    #('balance',
                    # TransformerWrapper(
                        #exclude = None,  
                        #include = None,
                        #transformer = FixImbalancer(
                            #estimator = SMOTE(#k_neighbors=5,
                            #n_jobs = None,
                            #random_state = None,
                            #sampling_strategy = 'auto',
                            # )))
                        #pp.Balancing_SMOTE_Encoding(
                            # label  = cfg.data_fields.label
                            # )), 

                    ('debbuging', pp.Debbuging()
                     ),


                    ('transformation', TransformerWrapper(
                        exclude = ['match'], 
                        include = None,
                        transformer = PowerTransformer(
                            #copy = False,
                            #method = 'yeo-johnson',
                            standardize = False
                            ))),

                    ('normalize', TransformerWrapper(
                        #exclude=['match'],
                        #include=None,
                        #transformer=StandardScaler(
                            #copy=False,
                            #with_mean=True,
                            # with_std=True
                            # ),
                        transformer=MinMaxScaler()
                        )),                                          
                    ],
            verbose=True)
        self.model = Pipeline(
            steps=[
                ('actual_estimator', VotingRegressor(
                    estimators=[
                        ('CatBoost Regressor', CatBoostRegressor(verbose=False,
                                                                 loss_function='RMSE'
                                                                 )),
                        ('Gradient Boosting Regressor', GradientBoostingRegressor(random_state=123)),
                        ('Huber Regressor', HuberRegressor(max_iter=300)),
                        ('Bayesian Ridge', BayesianRidge()),
                        ('Ridge Regression', Ridge(random_state=123))
                        ],
                    n_jobs=-1,
                    weights=[0.2, 0.2, 0.2, 0.2, 0.2]
                    ))])
   
    def makeDataPipeline(self):
        super().makeDataPipeline()
            
    def fit(self, score: dict) -> dict:
        super().fit(score)

    def predict(self, X: pd.DataFrame):
        super().predict(X)
