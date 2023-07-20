#model definition for the experimeent

from omegaconf import DictConfig

from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import HuberRegressor, BayesianRidge, Ridge

from pycaret.internal.pipeline import Pipeline
from pycaret.internal.preprocess.preprocessor import PowerTransformer, StandardScaler, SimpleImputer 
from pycaret.internal.preprocess.preprocessor import FixImbalancer, TransformerWrapper, TargetEncoder, OneHotEncoder, MinMaxScaler

from src.conf.di_f_models import Di_FX_Regressor
import src.conf.preprocessors as pp

from catboost import CatBoostRegressor
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import r2_score, mean_absolute_percentage_error


class MxRetailSalary1(Di_FX_Regressor):
    
    class Features(BaseModel):  # Rewritting Features class to include the actual features
        state: str = 'Hidalgo'
        income_employee_day: float = 4000.00
        employees_business: int = 6  
    
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        
        self.scores = [{'id': 'mape', 'metric': mean_absolute_percentage_error},
                       {'id': 'r2', 'metric': r2_score}]
        
        self.kfold = {'n_splits': 5, 'shuffle': True, 'random_state': self.cfg.general_ml.seed}
        
        print(f'Class: {self.di_fx[0]}-{self.di_fx[1]}-{self.di_fx[2]}')
        
        # here you define the datapipeline transformation model getting params from pycaret in data profiling (notebook)
        self.dataPipeline = Pipeline(   
            steps = [                 
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

                    ('debbuging', pp.Debbuging()
                     ),

                    ('transformation', 
                        TransformerWrapper(
                            exclude = ['match'], 
                            include = None,
                            transformer = PowerTransformer(
                                #copy = False,
                                #method = 'yeo-johnson',
                                standardize = False
                                ))),

                    ('normalize', 
                        TransformerWrapper(
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
        
        # and here you define the prediction model 
        self.model = Pipeline(
            steps=[
                (('CatBoost Regressor', CatBoostRegressor(verbose=False,
                                                                 loss_function='RMSE'
                                                                 )))])
   
    def runDataPipeline(self):
        super().runDataPipeline()
            
    def fit(self, tracking: bool) -> dict:
        super().fit(tracking)

    def fit_Kfold(self, tracking: bool = False) -> dict:
        super().fit_Kfold(tracking)

    def predict(self, X: pd.DataFrame):
        return super().predict(X)