# running the datapipeline transformation of the model

import hydra
from omegaconf import DictConfig
from src.FFNN_experiment_model import MxRetailSalary1


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    experiment = MxRetailSalary1(cfg)
    print(f'Created experiment:{experiment.id}')
    print(f'runing datapipeline transformation')
    experiment.runDataPipeline()    
    print(f'transformation concluded')
    
    print(experiment.catalogues)
    
    
if __name__ == "__main__":
    main()