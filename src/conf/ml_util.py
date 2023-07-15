#version: july15, 2023

import hydra
from omegaconf import DictConfig
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pipeline import datapipeline, datapipeline2, datapipeline3, datapipeline4


def load_data(pathfile: os.path, encoding:str='') -> pd.DataFrame:
    """
    to load data from raw directory and return a dataframe
    """
    print(f' STEP1: loading raw data...@ {pathfile}')
    df = pd.read_csv(pathfile, encoding=encoding)
    print(f'        Were loaded: (rows,cols) {df.shape}')
    print( '------------------------------------------------------')
    return df


def write_transformed(pathfile: os.path, data: pd.DataFrame)-> None:
     """
     saving transformed data in file
     """
     data.to_csv(pathfile, index=False)
     print(f'       Transformed data saved @ {pathfile}')
     print( '------------------------------------------------------')
     print('\nDTYPES:\n',data.dtypes)
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
                   seed: int
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
    
 
    print(' STEP3: Sppliting Data')

    df_train, df_test = train_test_split(data, test_size=percent_test, random_state=seed)
    df_train, df_validation = train_test_split(df_train, test_size= percent_valid, random_state=seed)
   
    print(f'       Once spplited: (rows,cols) in train set: {df_train.shape}') 
    print(f'                                  in validation set; {df_validation.shape}')
    print(f'                                  and in test set: {df_test.shape}')

    df_train[label].to_csv(pathfile_train_labels, index=None)
    df_train.drop(str(label), axis=1).to_csv(pathfile_train_futures, index=None)

    df_validation[label].to_csv(pathfile_validation_labels, index=None)
    df_validation.drop(str(label), axis=1).to_csv(pathfile_validation_futures, index=None)

    df_test[label].to_csv(pathfile_test_labels, index=None)
    df_test.drop(str(label), axis=1).to_csv(pathfile_test_futures, index=None)

    print(f'       Spplited data saved @ {pathfile_train_futures}')
    print(f'                             {pathfile_train_labels}')
    print(f'                             {pathfile_validation_futures}')
    print(f'                             {pathfile_validation_labels}')
    print(f'                             {pathfile_test_futures}')
    print(f'                             {pathfile_test_labels}')

