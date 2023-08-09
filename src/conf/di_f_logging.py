#  Logging definition file
#  version july 28, 2023

from omegaconf import DictConfig
import logging


di_f_logger = logging.getLogger(__name__)
di_f_logger.propagate = False  # This is to avoid terminal echo/propagation
di_f_logger.setLevel(
    level=logging.INFO
)  # From INFO, starting logging CRITICAL -> ERROR -> WARNING -> INFO -> DEBUG

formatter = logging.Formatter(
    "%(asctime)s - %(module)s - %(lineno)d - %(levelname)s - %(message)s"
)

# This handler is to write to the LOG file
# handler = logging.FileHandler(f'{cfg.paths.di_f_pipeline_log_dir}/{cfg.file_names.di_f_pipeline_log}')
handler = logging.FileHandler("logs/di_f_pipeline.log")

handler.setFormatter(formatter)
di_f_logger.addHandler(handler)

# And this one is to echo on screen
# comment folowing lines if don't want to output on screen
handler2 = logging.StreamHandler()
handler2.setFormatter(formatter)
di_f_logger.addHandler(handler2)
