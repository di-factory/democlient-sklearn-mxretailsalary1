import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import HuberRegressor, BayesianRidge, Ridge

from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import cross_val_score, KFold

from catboost import CatBoostRegressor

import os

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


regressor = Pipeline([('actual_estimator', VotingRegressor(
     estimators=[('CatBoost Regressor', CatBoostRegressor(verbose=False,
                                                          loss_function='RMSE')),
                 ('Gradient Boosting Regressor', GradientBoostingRegressor(random_state=123)),
                 ('Huber Regressor', HuberRegressor()),
                 ('Bayesian Ridge', BayesianRidge()),
                 ('Ridge Regression', Ridge(random_state=123))],
     n_jobs=-1,
     weights=[0.2, 0.2, 0.2, 0.2, 0.2]))
     ])

kfold = KFold(n_splits=5, shuffle=True, random_state=123)


def train_kfold(cfg: DictConfig, regressor: linear_model, kfold: KFold, score: dict)-> dict:
    # Load the data
    train_features = pd.read_csv(os.path.join(cfg.paths.processed_data_dir,
                                              cfg.file_names.train_features))
    train_labels = pd.read_csv(os.path.join(cfg.paths.processed_data_dir, 
                                            cfg.file_names.train_labels))

    validation_features = pd.read_csv(os.path.join(cfg.paths.processed_data_dir, 
                                                   cfg.file_names.validation_features))
    validation_labels = pd.read_csv(os.path.join(cfg.paths.processed_data_dir, 
                                                 cfg.file_names.validation_labels))

    test_features = pd.read_csv(os.path.join(cfg.paths.processed_data_dir, 
                                                   cfg.file_names.test_features))
    test_labels = pd.read_csv(os.path.join(cfg.paths.processed_data_dir, 
                                                 cfg.file_names.test_labels))

    #creating datasets from concatenation of sources
    #In this case as kfold, we need to concatenate the whole datasets
    whole_features = pd.concat([train_features, validation_features, test_features], ignore_index=True)
    whole_labels = pd.concat([train_labels, validation_labels, test_labels], ignore_index=True)
                                    
    #printing r2 sccore with cv=5
    sc = cross_val_score(regressor, whole_features, np.ravel(whole_labels), cv=5, scoring=score['id']).mean()
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

            regressor.fit(X_train, np.ravel(y_train))
            y_pred = regressor.predict(X_test)

            sc = score['metric'](y_test, y_pred)
            scores.append(sc)
   
    print(f'scoring after kfold: {np.mean(scores)}')
    return({'id':score['id'], 'result':np.mean(scores)})

    
def train(cfg : DictConfig)-> None:
    # Load the data
    train_features = pd.read_csv(os.path.join(cfg.paths.processed_data,
                                              cfg.file_names.train_features))
    train_labels = pd.read_csv(os.path.join(cfg.paths.processed_data, 
                                            cfg.file_names.train_labels))

    validation_features = pd.read_csv(os.path.join(cfg.paths.processed_data, 
                                                   cfg.file_names.validation_features))
    validation_labels = pd.read_csv(os.path.join(cfg.paths.processed_data, 
                                                 cfg.file_names.validation_labels))

    test_features = pd.read_csv(os.path.join(cfg.paths.processed_data, 
                                                   cfg.file_names.test_features))
    test_labels = pd.read_csv(os.path.join(cfg.paths.processed_data, 
                                                 cfg.file_names.test_labels))
    
    #test_labels.match = random.randint(0,1)

    mlflow.sklearn.autolog()

    for maxiter in [1000,1100]:
        with mlflow.start_run():
            # Create the model
            clf = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=maxiter,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=cfg.general_ml.seed, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)

            # Fit the model
            print('fitting the  model')
            print(train_features.shape, train_labels.shape)
            clf.fit(train_features, np.ravel(train_labels))

            # Make predictions

    
            print('making predictions')
            predictions = clf.predict(test_features)
            predict_proba = clf.predict_proba(test_features)
            signature = infer_signature(test_features, predictions)

            #for k, pred in enumerate(predictions):
            #print(f' {pred}, {predict_proba[k][0]:.6f},{predict_proba[k][1]:.6f} ')
            # Evaluate the model
            score= clf.score(test_features, test_labels)
            mlflow.log_metric("score", score)
            mlflow.sklearn.log_model(clf, 'MODEL',signature=signature)
            print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
        mlflow.end_run()

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig)-> None:
    #mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    #mlflow.set_experiment(cfg.mlflow.tracking_experiment_name)
    train_kfold(cfg, regressor, kfold,score= {'id':'r2', 'metric':r2_score})
    
if __name__ == "__main__":
    main()





