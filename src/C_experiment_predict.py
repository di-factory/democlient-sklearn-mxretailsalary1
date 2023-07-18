# running the datapipeline transformation of the model

import hydra
from omegaconf import DictConfig
from src.experiment_model import MxRetailSalary1
import pandas as pd


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    experiment = MxRetailSalary1(cfg)
    
    dummy_input = pd.DataFrame([['Oaxaca', 6021.07, 4],
                          ['Yucatan',34599.0, 4]], 
                         columns=experiment.features)    
    print('just dummy data:')
    experiment.predict(dummy_input)   

if __name__ == "__main__":
    main()