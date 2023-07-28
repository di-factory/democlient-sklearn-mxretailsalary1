# running the datapipeline transformation of the model

import hydra
from omegaconf import DictConfig
from src.experiment_model import MxRetailSalary1
from sklearn.metrics import r2_score, mean_absolute_percentage_error



@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    experiment = MxRetailSalary1(cfg)
    print(f'experiment.fit(): Experiment created:{experiment.id} class: {experiment.di_fx[0]}-{experiment.di_fx[1]}-{experiment.di_fx[2]}')
    print("experiment.fit(): Running")
    results = experiment.fit_Kfold(tracking=cfg.general_ml.tracking_on)    
    print(f'experiment.fit(): Trainning concluded, results: {results}')


if __name__ == "__main__":
    main()