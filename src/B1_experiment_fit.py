# running the datapipeline transformation of the model

import hydra
from omegaconf import DictConfig
from src.experiment_model import MxRetailSalary1



@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:


    experiment = MxRetailSalary1(cfg)
    print(f'Created experiment:{experiment.id}')
    print(f'runing datapipeline transformation')
    experiment.fit(tracking=cfg.general_ml.tracking_on)    
    print(f'trainning concluded')

if __name__ == "__main__":
    main()