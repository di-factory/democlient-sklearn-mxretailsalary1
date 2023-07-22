import hydra
from omegaconf import DictConfig
import torch
from src.conf.di_f_models import prueba_NN

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    experiment = prueba_NN(cfg, 34)
    print(f'Created experiment:{experiment.id}')
    print(f'runing datapipeline transformation')
    experiment.fit(tracking=cfg.general_ml.tracking_on)    
    print(f'trainning concluded')

if __name__ == "__main__":
    main()

