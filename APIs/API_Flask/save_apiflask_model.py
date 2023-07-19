# to save/serialize predict mode to be used by apiflask

import hydra
from omegaconf import DictConfig
from src.experiment_model import MxRetailSalary1
import pandas as pd
import joblib
import pydantic
import os


@hydra.main(version_base=None, config_path="../../src/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    
    experiment = MxRetailSalary1(cfg)
    input = pd.DataFrame([['Oaxaca', 6021.07, 4],
                          ['Yucatan', 34599.0, 4]], 
                         columns=experiment.feature_list
                         )
    
    print('just dummy data:')
    experiment.predict(input)   
    
    try:
        joblib.dump(experiment, os.path.join(cfg.paths.api_flask_dir, cfg.file_names.apiflask_predict_model))
        print("Api Flask model saved successfully!")
    except Exception as e:
        print(f"Error saving the API flask predict model: {e}")    
    

if __name__ == "__main__":
    main()