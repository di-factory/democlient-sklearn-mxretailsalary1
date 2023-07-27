# running the datapipeline transformation of the model

import hydra
from omegaconf import DictConfig
from src.FFNN_experiment_model import MxRetailSalary1


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    experiment = MxRetailSalary1(cfg)
    
    print(f'experiment.runDataPipeline(): Experiment created:{experiment.id} class: {experiment.di_fx[0]}-{experiment.di_fx[1]}-{experiment.di_fx[2]}')
    print("experiment.runDataPipeline(): Running")
    results = experiment.runDataPipeline(verbose = False)   
    print(f'experiment.fit(): Concluded, results: {results}')

    
if __name__ == "__main__":
    main()