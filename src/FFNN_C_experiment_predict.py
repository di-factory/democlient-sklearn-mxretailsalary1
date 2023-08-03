# running the datapipeline transformation of the model

import hydra
from omegaconf import DictConfig
from src.FFNN_experiment_model import MxRetailSalary1
import pandas as pd

from src.conf.di_f_logging import di_f_logger


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    experiment = MxRetailSalary1(cfg)
    di_f_logger.info(
        "--------------------------------------------------------------------------------------------------------"
    )
    di_f_logger.info(
        "------------------------- Starting process: Experiment_Prediction --------------------------------------"
    )

    dummy_input = pd.DataFrame(
        [["Oaxaca", 6021.07, 4], ["Yucatan", 34599.0, 4]],
        columns=experiment.feature_list,
    )
    di_f_logger.info(f"just dummy data to input: {dummy_input}")
    result = experiment.predict(dummy_input)
    di_f_logger.info(f"result: {result}")


if __name__ == "__main__":
    main()
