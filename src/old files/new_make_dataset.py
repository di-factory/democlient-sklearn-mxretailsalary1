#version: july 15, 2023

import hydra
from omegaconf import DictConfig
from src.conf.di_f_models import AnyExperiment
from sklearn.metrics import r2_score
import pandas as pd


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig)-> None:
    exp = AnyExperiment('prueba', cfg)
    #print(exp.id, exp.cfg)
    exp.makeDataPipeline()
    exp.fit_Kfold(score={'id': 'r2', 'metric': r2_score})
    input = pd.DataFrame([['Oaxaca', 6021.07, 4],
                          ['Yucatan',34599.0, 4]], 
                         columns=['state', 'income_employee_day', 'employees_business']
                         )
    #print(input)
    exp.predict(input)


if __name__ == "__main__":
    main()
