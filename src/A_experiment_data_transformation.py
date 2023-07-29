# running the datapipeline transformation of the model
from src.conf.di_f_logging import di_f_logger
import hydra
from omegaconf import DictConfig
from src.experiment_model import MxRetailSalary1

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    experiment = MxRetailSalary1(cfg)
    
    di_f_logger.info(f'experiment.runDataPipeline(): Experiment created:{experiment.id} class: {experiment.di_fx[0]}-{experiment.di_fx[1]}-{experiment.di_fx[2]}')
    di_f_logger.info("experiment.runDataPipeline(): Running")
    results = experiment.runDataPipeline()   
    di_f_logger.info(f'experiment.fit(): Concluded, results: {results}')

    
if __name__ == "__main__":
    main()