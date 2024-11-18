import logging
from colorlog import ColoredFormatter

log = logging.getLogger(__name__)


def init_logging(level:int, log_file:str, out_dir:str) -> None:
    if log_file == "":
        formatter = ColoredFormatter(
            "%(white)s%(asctime)10s | %(log_color)s%(levelname)6s | %(log_color)s%(message)6s",
            reset=True,
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'yellow',
                'WARNING':  'green',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            },
        )
        handler = logging.StreamHandler()
    else:
        formatter = logging.Formatter("%(asctime)10s | %(levelname)6s | %(message)6s")
        handler = logging.FileHandler(out_dir+ "/" + log_file+".txt", 'w')

    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(level)


def view_params(params, dataset_params, net_params):
    log.info("Dataset Parameters:")
    for k, v in dataset_params.items():
        log.info(f"{k}: {v}")
    log.info("Network Parameters:")
    for k, v in net_params.items():
        log.info(f"{k}: {v}")
    log.info("Training Parameters:")
    for k, v in params.items():
        log.info(f"{k}: {v}")