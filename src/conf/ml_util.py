#version: july15, 2023

import hydra
from omegaconf import DictConfig
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch

from pycaret.internal.pipeline import Pipeline as Pycaret_Pipeline
from pycaret.internal.preprocess.preprocessor import PowerTransformer, StandardScaler, SimpleImputer 
from pycaret.internal.preprocess.preprocessor import FixImbalancer, TransformerWrapper, TargetEncoder, OneHotEncoder, MinMaxScaler


from torch.utils.data import Dataset

import joblib


def load_data(pathfile: os.path, encoding:str='', verbose: bool = False) -> pd.DataFrame:
    """
    to load data from raw directory and return a dataframe
    """
    if verbose:
        print(f'load_data(): loading raw data...@ {pathfile}')
    df = pd.read_csv(pathfile, encoding=encoding)
    if verbose:
        print(f'        Were loaded: (rows,cols) {df.shape}')
        print( '------------------------------------------------------')
    return df


def write_transformed(pathfile: os.path, data: pd.DataFrame, verbose: bool = False)-> None:
     """
     saving transformed data in file
     """
     
     data.to_csv(pathfile, index=False)
     if verbose:
        print(f'write_transformed(): File were created @ {pathfile}')
        print( '------------------------------------------------------')
        print( '------------------------------------------------------')


def write_spplited(pathfile_train_futures: os.path,
                   pathfile_train_labels: os.path,
                   pathfile_validation_futures: os.path,
                   pathfile_validation_labels: os.path,
                   pathfile_test_futures: os.path,
                   pathfile_test_labels: os.path,
                   data: pd.DataFrame,
                   label: str,
                   percent_valid: float,
                   percent_test: float,
                   seed: int,
                   verbose: bool = False
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
    
 
    if verbose:
        print('write_splitted(): Split Files were created')

    df_train, df_test = train_test_split(data, test_size=percent_test, random_state=seed)
    df_train, df_validation = train_test_split(df_train, test_size= percent_valid, random_state=seed)
    
    if verbose:
        print(f'write_spplited():Once spplited: (rows,cols) in train set: {df_train.shape}') 
        print(f'                                  in validation set; {df_validation.shape}')
        print(f'                                  and in test set: {df_test.shape}')

    df_train[label].to_csv(pathfile_train_labels, index=None)
    df_train.drop(str(label), axis=1).to_csv(pathfile_train_futures, index=None)

    df_validation[label].to_csv(pathfile_validation_labels, index=None)
    df_validation.drop(str(label), axis=1).to_csv(pathfile_validation_futures, index=None)

    df_test[label].to_csv(pathfile_test_labels, index=None)
    df_test.drop(str(label), axis=1).to_csv(pathfile_test_futures, index=None)

    if verbose:
        print(f'write_spplited(): Spplited data saved @ {pathfile_train_futures}')
        print(f'                             {pathfile_train_labels}')
        print(f'                             {pathfile_validation_futures}')
        print(f'                             {pathfile_validation_labels}')
        print(f'                             {pathfile_test_futures}')
        print(f'                             {pathfile_test_labels}')


# Transformers: 

def minmax(values):  # This util, transform columns of a matrix between min-max and returns also the min/max values used in each column 
    minmax_cols = {} 
    for col in range(values.shape[1]):
        min_val = min(values[:,col])
        max_val = max(values[:,col])
        values[:,col] = (values[:,col] - min_val) / (max_val - min_val)
        minmax_cols[col]= {'min_val': min_val, 'max_val': max_val} 
    return minmax_cols, values
    
        
def normalize(values):  # This util, normalize columns of a matrix and returns also the mean/std values used in each column 
    normalize_cols = {} 
    for col in range(values.shape[1]):
        mean = values[:,col].mean()
        std = values[:,col].std()
        values[:,col]  = (values[:,col]  - mean) / std 
        normalize_cols[col]= {'mean': mean, 'max_val': std} 
    return normalize_cols, values


class PytorchDataSet(Dataset):
    def __init__(self):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
    
    
class Pycaret_DataPipeline():
    def __init__(self, cfg: DictConfig, data_pipeline: Pycaret_Pipeline= None):
        self.cfg = cfg
        self.dataPipeline = data_pipeline
        
    def fit(self, X, y):
        self.dataPipeline.fit(X, y)
    
    def transform(self, X):
        return self.dataPipeline.transform(X)
        
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
 
    
    
