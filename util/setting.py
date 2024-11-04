import logging
from colorlog import ColoredFormatter

log = logging.getLogger(__name__)


def init_logging(level:int, log_file:str) -> None:
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
        handler = logging.FileHandler("log/"+log_file+".txt", 'w')

    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(level)