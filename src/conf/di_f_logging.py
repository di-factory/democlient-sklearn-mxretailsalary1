import hydra
from omegaconf import DictConfig

import logging
import os


di_f_logger = logging.getLogger(__name__)
di_f_logger.propagate = False
di_f_logger.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(module)s - %(lineno)d - %(levelname)s - %(message)s')

handler = logging.FileHandler('logs/di_f_pipe.log')
handler.setFormatter(formatter)
di_f_logger.addHandler(handler)

handler2 = logging.StreamHandler()
handler2.setFormatter(formatter)
di_f_logger.addHandler(handler2)
