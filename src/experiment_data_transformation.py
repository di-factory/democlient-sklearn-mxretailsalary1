# running the datapipeline transformation of the model

import hydra
from omegaconf import DictConfig
from src.experiment_model import MxRetailSalary1



@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    experiment = MxRetailSalary1(cfg.general_ml.experiment, cfg)
    print(f'Created experiment:{experiment.id}')
    print(f'runing datapipeline transformation')
    experiment.runDataPipeline()    
    print(f'transformation concluded')

if __name__ == "__main__":
    main()