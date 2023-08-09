# version: july 15, 2023

from omegaconf import DictConfig

# from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
# from sklearn.linear_model import HuberRegressor, BayesianRidge, Ridge
# from sklearn.model_selection import train_test_split

from pycaret.internal.pipeline import Pipeline

# from pycaret.internal.preprocess.preprocessor import PowerTransformer, StandardScaler, SimpleImputer
# from pycaret.internal.preprocess.preprocessor import FixImbalancer, TransformerWrapper, TargetEncoder, OneHotEncoder, MinMaxScaler


from sklearn.model_selection import cross_val_score, KFold, train_test_split

from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    make_scorer,
)


# from catboost import CatBoostRegressor
import src.conf.di_f_datapipes as di_f_datapipes
import os
import pandas as pd
import numpy as np
import joblib

import mlflow
import mlflow.sklearn
from typing import List, Any
from pydantic import BaseModel

import torch

# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset  # ,Subset,SubsetRandomSampler

from src.conf.di_f_logging import di_f_logger


# from mlflow.models.signature import infer_signature


# --------- LEVEL 0 -----------------
class Di_F_Pipe:  # Main class for all the experiments definitions
    class Di_FX_score:  # To record the scores of each model
        def __init__(self, score: dict):
            self.id = score["id"]
            self.metric = score["metric"]

    class Di_FX_Kfold:  # To record the params of Kfolds
        def __init__(
            self,
            kfold_dict: dict = {"n_splits": 5, "shuffle": True, "random_state": 123},
        ):
            self.n_splits = kfold_dict["n_splits"]
            self.shuffle = kfold_dict["shuffle"]
            self.random_state = kfold_dict["random_state"]

    class Features(BaseModel):  # To represent the Features Set of the experiment
        pass

        def __repr__(self):
            return {"id": self.id, "metric": self.metric}

    def __init__(self, cfg: DictConfig):
        # def create_Features() -> DictConfig: # this codes keeps when  pkl can manage create_model from pydantic
        #    fields_dict = {}
        #    for record in self.cfg.data_fields.features:
        #        fields_dict[record.field] = (eval(record.type), record.default)
        #    return create_model('Features', **fields_dict)  # This retrurn a class

        def create_catalogues() -> (
            dict
        ):  # to save the catalogue of values of field according w config.yaml
            catalogue = {}
            for record in self.cfg.data_fields.features:
                if hasattr(
                    record, "aList"
                ):  # Looking for the field aList in each record
                    catalogue[record.field] = record.aList
            return catalogue

        self.di_fx: List[str] = []
        self.di_fx.append("Di_F_Pipe")  # Level 0 class of the experiment

        self.cfg: DictConfig = cfg  # config.yaml file
        self.id: str = self.cfg.general_ml.experiment  # id = client.project.experiment
        self.features: self.Features  # create_Features()()  # Features as instance of BaseModel by create_features
        self.feature_list = [r.field for r in self.cfg.data_fields.features]

        self.dataPipeline: Pipeline = None  # Data Pipeline
        self.model: Any = None  # ML Pipeline

        self.catalogues: dict = create_catalogues()
        self.scores: List[self.Di_FX_score] = None
        self.kfold = self.Di_FX_Kfold()

    def load_model(
        self,
    ) -> (
        None
    ):  # this method loads the model prediction that was created/saved in fits methods
        pass

    def save_model(self) -> None:  # this method saves the model prediction
        pass

    def runDataPipeline(self) -> dict:  # this method runs the dataPipeline object-class
        pass

    def fit(
        self, tracking: bool = False
    ) -> dict:  # this methods train the model prediction defined.
        pass

    def fit_Kfold(self) -> dict:  # this method use Crossvalidations in trainning
        pass

    def predict(
        self, X: pd.DataFrame
    ) -> np.array:  # this method makes predictions of unseen incomig data
        pass

    def evaluate(
        self,
    ) -> (
        dict
    ):  # this method evaluate the tarinned model with unseen data of test dataset.
        pass


# --------- LEVEL 1 ------Type of experiment: Regression, Classification, etc-----------
class Di_F_Pipe_Regression(Di_F_Pipe):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.di_fx.append("Regression")  # Level 1 class of the experiment

    def runDataPipeline(self) -> dict:  # this method runs the dataPipeline object-class
        pass

    def fit(
        self, tracking: bool = False
    ) -> dict:  # this methods train the model prediction defined.
        pass

    def fit_Kfold(
        self, tracking: bool = False
    ) -> dict:  # this method use Crossvalidations in trainning
        pass

    def predict(
        self, X: pd.DataFrame
    ) -> np.array:  # this method makes predictions of unseen incomig data
        pass

    def evaluate(
        self,
    ) -> (
        dict
    ):  # this method evaluate the tarinned model with unseen data of test dataset.
        pass


class Di_F_Pipe_Classification(Di_F_Pipe):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.di_fx.append("Classification")  # Level 1 class of the experiment

    def runDataPipeline(self) -> None:  # this method runs the dataPipeline object-class
        pass

    def fit(
        self, tracking: bool = False
    ) -> dict:  # this methods train the model prediction defined.
        pass

    def fit_Kfold(
        self, tracking: bool = False
    ) -> dict:  # this method use Crossvalidations in trainning
        pass

    def predict(
        self, X: pd.DataFrame
    ) -> np.array:  # this method makes predictions of unseen incomig data
        pass

    def evaluate(
        self,
    ) -> (
        dict
    ):  # this method evaluate the tarinned model with unseen data of test dataset.
        pass


# --------- LEVEL 2 ---------- platform for solution: sklearn, pytorch,, Tensorflow, etc.
class Di_F_Pipe_Regression_Pycaret(Di_F_Pipe_Regression):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.di_fx.append("Pycaret")  # Level 1 class of the experiment

    def load_model(
        self,
    ) -> (
        None
    ):  # this method loads the model prediction that was created/saved in fits methods
        try:
            self.model = joblib.load(
                os.path.join(self.cfg.paths.models_dir, self.cfg.file_names.model)
            )
            di_f_logger.info("model loaded successfully!")
        except Exception as e:
            di_f_logger.info(f"Error loading the model: {e}")

    def save_model(self) -> None:  # this method saves the model prediction
        try:
            joblib.dump(
                self.model,
                os.path.join(self.cfg.paths.models_dir, self.cfg.file_names.model),
            )
            di_f_logger.info("model saved successfully!")
        except Exception as e:
            di_f_logger.info(f"Error saving the model: {e}")

    def runDataPipeline(self) -> dict:  # this method runs the dataPipeline object-class
        result = {}  # To grab results

        # loading pre-processed data from notebook data profiling
        pathfile_l = os.path.join(
            self.cfg.paths.processed_data_dir, self.cfg.file_names.processed_data
        )
        data = di_f_datapipes.load_data(
            pathfile_l
            # encoding=<particular encoding>
        )

        # Sppliting the data

        train_data, test_data = train_test_split(
            data,
            test_size=self.cfg.data_pipeline.data_transform_params.percent_test,
            random_state=self.cfg.general_ml.seed,
        )
        train_data, val_data = train_test_split(
            train_data,
            test_size=self.cfg.data_pipeline.data_transform_params.percent_valid,
            random_state=self.cfg.general_ml.seed,
        )

        # executing trainning transformation
        result["initial_trainning_shape"] = train_data.shape

        di_f_logger.info("runDataPipeline(): Transforming train data init")

        train_labels = train_data[self.cfg.data_fields.label]
        self.dataPipeline.fit(
            train_data[self.feature_list], train_labels
        )  # only fitting features, no label

        train_data = self.dataPipeline.transform(
            train_data[self.feature_list]
        )  # transformations is making on features

        di_f_logger.info(
            f"runDataPipeline(): After train transformation: (rows,cols) {train_data.shape}"
        )
        result["final_train_data_shape"] = train_data.shape

        # executing validation transformation
        result["initial_validation_data_shape"] = val_data.shape

        di_f_logger.info("runDataPipeline(): Transforming validation data init")

        val_labels = val_data[self.cfg.data_fields.label]

        val_data = self.dataPipeline.transform(
            val_data[self.feature_list]
        )  # transformations is making on features

        di_f_logger.info(
            f"runDataPipeline(): After val data transformation: (rows,cols) {val_data.shape}"
        )
        result["final_validation_data_shape"] = val_data.shape

        # executing testing (unseen) data transformation
        result["initial testing_data_shape"] = test_data.shape

        di_f_logger.info("runDataPipeline(): Transforming testing data init")

        test_labels = test_data[self.cfg.data_fields.label]

        test_data = self.dataPipeline.transform(
            test_data[self.feature_list]
        )  # transformations is making on features

        di_f_logger.info(
            f"runDataPipeline(): After test data transformation: (rows,cols) {test_data.shape}"
        )
        result["final_test_data_shape"] = test_data.shape

        # setting the directory where to save transformed data
        pathfile_s = os.path.join(
            self.cfg.paths.interim_data_dir, self.cfg.file_names.data_file
        )

        # write the whole transformed data
        data = pd.concat([train_data, val_data, test_data], axis=0)
        di_f_datapipes.write_transformed(pathfile_s, data)

        # setting paths where to save splitted data
        pathfile_train_futures = os.path.join(
            self.cfg.paths.processed_data_dir, self.cfg.file_names.train_features
        )
        pathfile_train_labels = os.path.join(
            self.cfg.paths.processed_data_dir, self.cfg.file_names.train_labels
        )
        pathfile_validation_futures = os.path.join(
            self.cfg.paths.processed_data_dir, self.cfg.file_names.validation_features
        )
        pathfile_validation_labels = os.path.join(
            self.cfg.paths.processed_data_dir, self.cfg.file_names.validation_labels
        )
        pathfile_test_futures = os.path.join(
            self.cfg.paths.processed_data_dir, self.cfg.file_names.test_features
        )
        pathfile_test_labels = os.path.join(
            self.cfg.paths.processed_data_dir, self.cfg.file_names.test_labels
        )

        # executting and writting spplited data
        di_f_datapipes.write_spplited(
            pathfile_train_futures,
            pathfile_train_labels,
            pathfile_validation_futures,
            pathfile_validation_labels,
            pathfile_test_futures,
            pathfile_test_labels,
            train_data,
            train_labels,
            val_data,
            val_labels,
            test_data,
            test_labels,
        )

        # save data pipeline model
        self.dataPipeline.save_dataPipeline()
        return result

    def fit(
        self, tracking: bool = False
    ) -> dict:  # this methods train the model prediction defined.
        # Load the data
        train_features = pd.read_csv(
            os.path.join(
                self.cfg.paths.processed_data_dir, self.cfg.file_names.train_features
            )
        )
        train_labels = pd.read_csv(
            os.path.join(
                self.cfg.paths.processed_data_dir, self.cfg.file_names.train_labels
            )
        )
        validation_features = pd.read_csv(
            os.path.join(
                self.cfg.paths.processed_data_dir,
                self.cfg.file_names.validation_features,
            )
        )
        validation_labels = pd.read_csv(
            os.path.join(
                self.cfg.paths.processed_data_dir, self.cfg.file_names.validation_labels
            )
        )
        test_features = pd.read_csv(
            os.path.join(
                self.cfg.paths.processed_data_dir, self.cfg.file_names.test_features
            )
        )
        test_labels = pd.read_csv(
            os.path.join(
                self.cfg.paths.processed_data_dir, self.cfg.file_names.test_labels
            )
        )

        # creating datasets from concatenation of sources
        # In this case as kfold, we need to concatenate the whole datasets
        X_train = pd.concat([train_features, validation_features], ignore_index=True)
        y_train = pd.concat([train_labels, validation_labels], ignore_index=True)

        if tracking:
            # setting up mlflow
            mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
            mlflow.set_experiment(
                f"{self.cfg.mlflow.tracking_experiment_name}_{self.di_f_exp}"
            )
            mlflow.sklearn.autolog()
            mlflow.start_run()

        # Fit the model
        di_f_logger.info("fitting the  model")
        di_f_logger.info(
            f"dimensions: features-> {X_train.shape}, labels-> {y_train.shape}"
        )

        self.model.fit(X_train, np.ravel(y_train))

        # Initializing dict results
        results = {}
        scores = []

        # calculating trainning scores
        y_pred = self.model.predict(X_train)

        di_f_logger.info("scores for trainning:")
        for score in self.scores:
            sc = score["metric"](y_train, y_pred)
            scores.append((score["id"], sc))

            di_f_logger.info(f"train scoring {score['id']}: {sc}")
        results["train"] = scores

        # calculating testing scores
        scores = []
        y_pred = self.model.predict(test_features)

        di_f_logger.info("scores for test:")
        for score in self.scores:
            sc = score["metric"](test_labels, y_pred)
            scores.append((score["id"], sc))

            di_f_logger.info(f"test scoring {score['id']}: {sc}")
        results["test"] = scores

        # saving the model
        self.save_model()

        if tracking:
            # stoping mlflow run experiment
            mlflow.end_run()

        return results

    def fit_Kfold(
        self, tracking: bool = False
    ) -> dict:  # this method use Crossvalidations in trainning
        # Set variable verbose to activate prints in function
        verbose = self.cfg.ml_pipeline.verbose

        # Load the data
        train_features = pd.read_csv(
            os.path.join(
                self.cfg.paths.processed_data_dir, self.cfg.file_names.train_features
            )
        )
        train_labels = pd.read_csv(
            os.path.join(
                self.cfg.paths.processed_data_dir, self.cfg.file_names.train_labels
            )
        )
        validation_features = pd.read_csv(
            os.path.join(
                self.cfg.paths.processed_data_dir,
                self.cfg.file_names.validation_features,
            )
        )
        validation_labels = pd.read_csv(
            os.path.join(
                self.cfg.paths.processed_data_dir, self.cfg.file_names.validation_labels
            )
        )
        test_features = pd.read_csv(
            os.path.join(
                self.cfg.paths.processed_data_dir, self.cfg.file_names.test_features
            )
        )
        test_labels = pd.read_csv(
            os.path.join(
                self.cfg.paths.processed_data_dir, self.cfg.file_names.test_labels
            )
        )

        # creating the kfold
        kfold = KFold(
            n_splits=self.kfold["n_splits"],
            shuffle=self.kfold["shuffle"],
            random_state=self.cfg.general_ml.seed,
        )

        # creating datasets from concatenation of sources
        # In this case as kfold, we need to concatenate the whole datasets
        kfold_features = pd.concat(
            [train_features, validation_features], ignore_index=True
        )
        kfold_labels = pd.concat([train_labels, validation_labels], ignore_index=True)

        if tracking:
            # setting up mlflow
            mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
            mlflow.set_experiment(
                f"{self.cfg.mlflow.tracking_experiment_name}_{self.di_f_exp}"
            )
            mlflow.sklearn.autolog()
            mlflow.start_run()

        # Initializing dict results, scores and scores_v
        results = {}
        scores = []

        scores_v = {}  # this dict is to grab the different metrics
        for score in self.scores:  # Initializing vectors to cumpute scores in each fold
            scores_v[score["id"]] = 0
        if verbose:
            di_f_logger.info("scores pre-Kfold:")

        for score in self.scores:
            sc = cross_val_score(
                self.model,
                kfold_features,
                np.ravel(kfold_labels),
                cv=5,
                scoring=make_scorer(score["metric"]),
            ).mean()
            scores.append((score["id"], sc))
            if verbose:
                di_f_logger.info(f"cross_val_scoring (before kfold){score['id']}: {sc}")
        results["cross_val_score"] = scores

        # Fit the model
        if verbose:
            di_f_logger.info("fitting the  model")
            di_f_logger.info(
                f"dimensions: features-> {kfold_features.shape}, labels-> {kfold_labels.shape}"
            )

        for train_index, test_index in kfold.split(kfold_features):
            X_train = np.array(kfold_features)[train_index]
            X_val = np.array(kfold_features)[test_index]
            y_train = np.array(kfold_labels)[train_index]
            y_val = np.array(kfold_labels)[test_index]

            self.model.fit(X_train, np.ravel(y_train))
            y_pred = self.model.predict(X_val)

            for score in self.scores:
                sc = score["metric"](y_val, y_pred)
                scores_v[score["id"]] += sc

        if verbose:
            di_f_logger.info("scores post-Kfold:")

        scores = []
        for score in self.scores:
            if verbose:
                di_f_logger.info(
                    f"scoring after kfold {score['id']}: {scores_v[score['id']]/self.kfold['n_splits']}"
                )
            scores.append((score["id"], sc))
        results["train"] = scores

        # Evaluating the model with unseen data
        if verbose:
            di_f_logger.info("Model evaluation:")

        X_test = np.array(test_features)
        y_test = np.array(test_labels)

        y_pred = self.model.predict(X_test)

        scores = []
        y_pred = self.model.predict(X_test)

        if verbose:
            di_f_logger.info("scores for test:")

        for score in self.scores:
            sc = score["metric"](y_test, y_pred)
            scores.append((score["id"], sc))
            if verbose:
                di_f_logger.info(f"test scoring {score['id']}: {sc}")
        results["test"] = scores

        # saving the model
        self.save_model()

        if tracking:
            # stoping mlflow run experiment
            mlflow.end_run()

        return results

    def predict(
        self, X: pd.DataFrame
    ) -> np.array:  # this method makes predictions of unseen incomig data
        di_f_logger.info(f"Loading data pipeline and ml pipeline trainned models")
        self.dataPipeline.load_dataPipeline()
        self.load_model()
        di_f_logger.info(f"predicting a vector of {X.shape}")
        X_transformed = self.dataPipeline.transform(X)
        result = self.model.predict(X_transformed)

        return np.array(result)

    def evaluate(
        self,
    ) -> (
        dict
    ):  # this method evaluate the tarinned model with unseen data of test dataset.
        pass


class Di_F_Pipe_Regression_Pytorch(Di_F_Pipe_Regression):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.di_fx.append("Pytorch")  # Level 1 class of the experiment

    def load_model(
        self,
    ) -> (
        None
    ):  # this method loads the model prediction that was created/saved in fits methods
        try:
            self.model.load_state_dict(
                torch.load(
                    os.path.join(self.cfg.paths.models_dir, self.cfg.file_names.model)
                )
            )
            di_f_logger.info("model loaded successfully!")
        except Exception as e:
            di_f_logger.info(f"Error loading the model: {e}")

    def save_model(self) -> None:  # this method saves the model prediction
        try:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.cfg.paths.models_dir, self.cfg.file_names.model),
            )
            di_f_logger.info("model saved successfully!")
        except Exception as e:
            di_f_logger.info(f"Error saving the model: {e}")

    def runDataPipeline(self) -> dict:  # this method runs the dataPipeline object-class
        result = {}  # To grab results

        # loading pre-processed data from notebook data profiling
        pathfile_l = os.path.join(
            self.cfg.paths.processed_data_dir, self.cfg.file_names.processed_data
        )
        data = di_f_datapipes.load_data(
            pathfile_l
            # encoding=<particular encoding>
        )

        # Sppliting the data

        train_data, test_data = train_test_split(
            data,
            test_size=self.cfg.data_pipeline.data_transform_params.percent_test,
            random_state=self.cfg.general_ml.seed,
        )
        train_data, val_data = train_test_split(
            train_data,
            test_size=self.cfg.data_pipeline.data_transform_params.percent_valid,
            random_state=self.cfg.general_ml.seed,
        )

        # executing trainning transformation
        result["initial_trainning_shape"] = train_data.shape

        di_f_logger.info("runDataPipeline(): Transforming train data init")

        train_labels = train_data[self.cfg.data_fields.label]
        self.dataPipeline.fit(
            train_data[self.feature_list], train_labels
        )  # only fitting features, no label

        train_data = self.dataPipeline.transform(
            train_data[self.feature_list]
        )  # transformations is making on features

        di_f_logger.info(
            f"runDataPipeline(): After train transformation: (rows,cols) {train_data.shape}"
        )
        result["final_train_data_shape"] = train_data.shape

        # executing validation transformation
        result["initial_validation_data_shape"] = val_data.shape

        di_f_logger.info("runDataPipeline(): Transforming validation data init")

        val_labels = val_data[self.cfg.data_fields.label]

        val_data = self.dataPipeline.transform(
            val_data[self.feature_list]
        )  # transformations is making on features

        di_f_logger.info(
            f"runDataPipeline(): After val data transformation: (rows,cols) {val_data.shape}"
        )
        result["final_validation_data_shape"] = val_data.shape

        # executing testing (unseen) data transformation
        result["initial testing_data_shape"] = test_data.shape

        di_f_logger.info("runDataPipeline(): Transforming testing data init")

        test_labels = test_data[self.cfg.data_fields.label]

        test_data = self.dataPipeline.transform(
            test_data[self.feature_list]
        )  # transformations is making on features

        di_f_logger.info(
            f"runDataPipeline(): After test data transformation: (rows,cols) {test_data.shape}"
        )
        result["final_test_data_shape"] = test_data.shape

        # setting the directory where to save transformed data
        pathfile_s = os.path.join(
            self.cfg.paths.interim_data_dir, self.cfg.file_names.data_file
        )

        # write the whole transformed data
        data = pd.concat([train_data, val_data, test_data], axis=0)
        di_f_datapipes.write_transformed(pathfile_s, data)

        # setting paths where to save splitted data
        pathfile_train_futures = os.path.join(
            self.cfg.paths.processed_data_dir, self.cfg.file_names.train_features
        )
        pathfile_train_labels = os.path.join(
            self.cfg.paths.processed_data_dir, self.cfg.file_names.train_labels
        )
        pathfile_validation_futures = os.path.join(
            self.cfg.paths.processed_data_dir, self.cfg.file_names.validation_features
        )
        pathfile_validation_labels = os.path.join(
            self.cfg.paths.processed_data_dir, self.cfg.file_names.validation_labels
        )
        pathfile_test_futures = os.path.join(
            self.cfg.paths.processed_data_dir, self.cfg.file_names.test_features
        )
        pathfile_test_labels = os.path.join(
            self.cfg.paths.processed_data_dir, self.cfg.file_names.test_labels
        )

        # executting and writting spplited data
        di_f_datapipes.write_spplited(
            pathfile_train_futures,
            pathfile_train_labels,
            pathfile_validation_futures,
            pathfile_validation_labels,
            pathfile_test_futures,
            pathfile_test_labels,
            train_data,
            train_labels,
            val_data,
            val_labels,
            test_data,
            test_labels,
        )

        # save data pipeline model
        self.dataPipeline.save_dataPipeline()
        return result

    def fit(
        self, tracking: bool = False
    ) -> dict:  # this methods train the model prediction defined.
        # Loadibg the data in 6 sets
        train_features = pd.read_csv(
            os.path.join(
                self.cfg.paths.processed_data_dir, self.cfg.file_names.train_features
            )
        )
        train_labels = pd.read_csv(
            os.path.join(
                self.cfg.paths.processed_data_dir, self.cfg.file_names.train_labels
            )
        )
        validation_features = pd.read_csv(
            os.path.join(
                self.cfg.paths.processed_data_dir,
                self.cfg.file_names.validation_features,
            )
        )
        validation_labels = pd.read_csv(
            os.path.join(
                self.cfg.paths.processed_data_dir, self.cfg.file_names.validation_labels
            )
        )
        test_features = pd.read_csv(
            os.path.join(
                self.cfg.paths.processed_data_dir, self.cfg.file_names.test_features
            )
        )
        test_labels = pd.read_csv(
            os.path.join(
                self.cfg.paths.processed_data_dir, self.cfg.file_names.test_labels
            )
        )

        # changing from datasets to tensors
        train_features = torch.tensor(train_features.values, dtype=torch.float)
        train_labels = torch.log(
            torch.tensor(train_labels.values, dtype=torch.float)
        )  # scaling to log the labels
        validation_features = torch.tensor(
            validation_features.values, dtype=torch.float
        )
        validation_labels = torch.log(
            torch.tensor(validation_labels.values, dtype=torch.float)
        )  # scaling to log the labels
        test_features = torch.tensor(test_features.values, dtype=torch.float)
        test_labels = torch.log(
            torch.tensor(test_labels.values, dtype=torch.float)
        )  # scaling to log the labels

        # creating datasets and dataloaders
        train_data = di_f_datapipes.PytorchDataSet(X=train_features, y=train_labels)
        validation_data = di_f_datapipes.PytorchDataSet(
            validation_features, validation_labels
        )

        train_loader = DataLoader(
            train_data, batch_size=self.model.batch_size, shuffle=True, drop_last=True
        )
        validation_loader = DataLoader(
            validation_data,
            batch_size=self.model.batch_size,
            shuffle=True,
            drop_last=True,
        )

        if tracking:
            # setting up mlflow
            mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
            mlflow.set_experiment(
                f"{self.cfg.mlflow.tracking_experiment_name}_{self.di_f_exp}"
            )
            mlflow.sklearn.autolog()
            mlflow.start_run()

        # Fit the model
        di_f_logger.info("fitting the  model")
        di_f_logger.info(
            f"dimensions: features-> {train_features.shape}, labels-> {train_labels.shape}"
        )

        # Initializing dict results, scores and scores_v
        results = {}

        scores_v = {}  # this dict is to grab the different validation metrics
        scores_t = {}  # this dict is to grab the different testing metrics

        for score in self.scores:  # Initializing vectors to cumpute scores in each fold
            scores_v[score["id"]] = 0
            scores_t[score["id"]] = 0

        self.losses = torch.zeros(
            self.model.num_epochs
        )  # To grab the loss_train for each epoch an see evolution

        for e in range(self.model.num_epochs):
            # switching to train mode
            self.model.train()
            batch_loss = []  # To grab the loss_train for each batch and then get mean()

            for X, y in train_loader:  # steping by each pair X,y of size batch_size
                y_pred = self.model.forward(X)

                loss = self.model.loss_func(y_pred, y)

                # backpropagation block
                self.model.optimizer.zero_grad()  # reinit gradients
                loss.backward()
                self.model.optimizer.step()
                batch_loss.append(loss.item())

                for score in self.scores:  # Adding each metric for each epoch
                    sc = score["metric"](y, y_pred.detach().numpy())
                    scores_t[score["id"]] += sc

            self.losses[e] = np.mean(batch_loss)
            """
            for score in self.scores:  # geting mean of each metric for each epoch
                scores.append((score["id"], scores_t[score["id"]] / self.model.num_epochs))
                results["train"] = scores
            """
            # let's evaluate:
            self.model.eval()

            X, y = next(iter(validation_loader))  # Get the validation set in one step

            with torch.no_grad():  # stop gradient descent in eval)
                y_pred = self.model.forward(X)

                for score in self.scores:  # Adding each metric for each epoch
                    sc = score["metric"](y, y_pred)
                    scores_v[score["id"]] += sc

                if (e + 1) % 100 == 0:
                    di_f_logger.info(f"epoch:{e+1}, loss:{self.losses[e]}")

        scores = []
        print(len(train_loader))
        for score in self.scores:  # geting mean of each metric for each epoch
            scores.append(
                (
                    score["id"],
                    scores_t[score["id"]] / (self.model.num_epochs * len(train_loader)),
                )
            )
        results["trainning"] = scores

        scores = []
        for score in self.scores:  # geting mean of each metric for each epoch
            scores.append((score["id"], scores_v[score["id"]] / self.model.num_epochs))
        results["validation"] = scores

        # Claculating metrics for testing
        self.model.eval()
        scores = []
        y_pred = self.model.forward(test_features)

        di_f_logger.info("scores for test:")

        for score in self.scores:
            sc = score["metric"](test_labels.detach().numpy(), y_pred.detach().numpy())
            scores.append((score["id"], sc))

            di_f_logger.info(f"test scoring {score['id']}: {sc}")
        results["test"] = scores

        # saving the model
        self.save_model()

        if tracking:
            # stoping mlflow run experiment
            mlflow.end_run()

        return results

    def fit_Kfold(
        self, tracking: bool = False
    ) -> dict:  # this method use Crossvalidations in trainning
        pass

    def predict(
        self, X: pd.DataFrame
    ) -> np.array:  # this method makes predictions of unseen incomig data
        di_f_logger.info(f"Loading data pipeline and ml pipeline trainned models")

        self.dataPipeline.load_dataPipeline()
        self.load_model()

        di_f_logger.info(f"predicting a vector of {X.shape}")

        X_transformed = self.dataPipeline.transform(X)

        result = self.model.forward(
            torch.tensor(X_transformed.values, dtype=torch.float)
        )
        result = torch.exp(result).detach().numpy()

        return np.array(result)

    def evaluate(
        self,
    ) -> (
        dict
    ):  # this method evaluate the tarinned model with unseen data of test dataset.
        pass


# --------- LEVEL 3 ---------- technology: model + datapipe to solve experiment
class Di_F_Pipe_Regression_Pycaret_Voating(Di_F_Pipe_Regression_Pycaret):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.di_fx.append("Voting")  # Level 2 class of the experiment

    def runDataPipeline(self) -> dict:
        result = super().runDataPipeline()
        return result

    def fit(
        self, tracking: bool = False
    ) -> dict:  # this methods train the model prediction defined.
        result = super().fit(tracking)
        return result

    def fit_Kfold(
        self, tracking: bool = False
    ) -> dict:  # this method use Crossvalidations in trainning
        result = super().fit_Kfold(tracking)
        return result

    def predict(
        self, X: pd.DataFrame
    ) -> np.array:  # this method makes predictions of unseen incomig data
        result = super().predict(X)
        return result


class Di_F_Pipe_Regression_Pytorch_FFNN(Di_F_Pipe_Regression_Pytorch):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.di_fx.append("FFNN")  # Level 2 class of the experiment

    def runDataPipeline(self) -> dict:
        result = super().runDataPipeline()
        return result

    def fit(
        self, tracking: bool = False
    ) -> dict:  # this methods train the model prediction defined.
        result = super().fit(tracking)
        return result

    def fit_Kfold(
        self, tracking: bool = False
    ) -> dict:  # this method use Crossvalidations in trainning
        pass

    def predict(
        self, X: pd.DataFrame
    ) -> np.array:  # this method makes predictions of unseen incomig data
        result = super().predict(X)
        return result


# -------------------------------------------- end
