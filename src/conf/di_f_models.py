#version: july 15, 2023

from omegaconf import DictConfig
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import HuberRegressor, BayesianRidge, Ridge

from pycaret.internal.pipeline import Pipeline
from pycaret.internal.preprocess.preprocessor import PowerTransformer, StandardScaler, SimpleImputer 
from pycaret.internal.preprocess.preprocessor import FixImbalancer, TransformerWrapper, TargetEncoder, OneHotEncoder, MinMaxScaler


import src.conf.preprocessors as pp
from sklearn.model_selection import cross_val_score, KFold

from catboost import CatBoostRegressor
import src.conf.ml_util as ml_util
import os
import pandas as pd
import numpy as np
import joblib

import mlflow
import mlflow.sklearn


from pydantic import BaseModel


#from mlflow.models.signature import infer_signature


class Di_F_Experiment:  # Main class for all the experiments definitions

    class Di_F_Experiment_score():
        def __init__(self, score: dict):
            self.id = score['id']
            self.metric = score['metric']
    
    class Features(BaseModel):
        pass
    
        def __repr__(self):
            return {'id': self.id, 'metric': self.metric}

    def __init__(self, cfg: DictConfig):
        self.di_f_exp: str  # class of the experiment 
        self.cfg: DictConfig = cfg  # config.yaml file 
        
        self.id: str = self.cfg.general_ml.experiment  # id = client.project.experiment
        self.features: self.Features = self.cfg.data_fields.features  # Features in config.yaml file
        
        self.dataPipeline: Pipeline = None  # Data Pipeline
        self.model: Pipeline = None  # ML Pipeline

  
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

    def fit(self, score: Di_F_Experiment_score, tracking: bool = False) -> dict:  # this methods train the model prediction defined.
        pass

    def fit_Kfold(self,  score: Di_F_Experiment_score, 
                  kfold_dict: dict = {'n_splits': 5, 'shuffle': True, 'random_state': 123},) -> dict:  # this method use Crossvalidations in trainning
        pass

    def predict(self, X: pd.DataFrame) -> np.array:  # this method makes predictions of unseen incomig data 
        pass


class Di_F_Experiment_Regressor(Di_F_Experiment):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.di_f_exp = 'VotingRegressor'

    def runDataPipeline(self) -> None:  # this method runs the dataPipeline object-class

        def transform_data(data: pd.DataFrame) -> pd.DataFrame:
            """
            to transform data using a sklearn pipeline technology 
            data is the whole dataset, fetures + label
            """
            print(' STEP2: Transforming data using a PIPELINE')
   
            dpl = self.dataPipeline  # instanciating dataPipeline object
            labels = data[self.cfg.data_fields.label]
            dpl.fit(data[self.cfg.data_fields.features], labels)  # only fitting features, no label
            df = dpl.transform(data[self.cfg.data_fields.features])  # transformations is making on features
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

    def fit(self, score: Di_F_Experiment.Di_F_Experiment_score, tracking: bool = False) -> dict:  # this methods train the model prediction defined.
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
        sc = score['metric'](y_train, y_pred)
        print(f'train scoring: {sc}')
        

        #calculating testing score
        y_pred = self.model.predict(test_features)
        sc = score['metric'](test_labels, y_pred)
        print(f'test scoring: {sc}')
        
        # saving the model
        self.save_model()
        
        if tracking:
            # stoping mlflow run experiment
            mlflow.end_run()
        
        return {'id': score['id'], 'result': sc}

    def fit_Kfold(self,  score: Di_F_Experiment.Di_F_Experiment_score, 
                  kfold_dict: dict = {'n_splits': 5, 'shuffle': True, 'random_state': 123},
                  tracking: bool = False) -> dict:  # this method use Crossvalidations in trainning
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
        sc = cross_val_score(self.model, whole_features, np.ravel(whole_labels), cv=5, scoring=score['id']).mean()
        print(f'cross_val_scoring (before kfold): {sc}')

        # Fit the model
        print('fitting the  model')
        print(f'dimensions: features-> {whole_features.shape}, labels-> {whole_labels.shape}')
   
        scores=[]

    
        for train_index, test_index in kfold.split(whole_features):
           
            X_train= np.array(whole_features)[train_index]
            X_test = np.array(whole_features)[test_index]
            y_train = np.array(whole_labels)[train_index]
            y_test = np.array(whole_labels)[test_index]

            self.model.fit(X_train, np.ravel(y_train))
            y_pred = self.model.predict(X_test)

            sc = score['metric'](y_test, y_pred)
            scores.append(sc)
   
        print(f'scoring after kfold: {np.mean(scores)}')
        
        # saving the model
        self.save_model()
        
        if tracking:
            # stoping mlflow run experiment
            mlflow.end_run()
        
        return {'id': score['id'], 'result': np.mean(scores)}

    def predict(self, X: pd.DataFrame) -> np.array:  # this method makes predictions of unseen incomig data 
        #print(X.head(), X.dtypes)
        self.load_dataPipeline()
        self.load_model()
        
        X_transformed = self.dataPipeline.transform(X)
        #print(X_transformed)
        result = self.model.predict(X_transformed)
        print(result)
        return np.array(result)


class Metaclass(Di_F_Experiment):  # for using templates
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
