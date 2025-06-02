import logging
import sys
from colorama import Fore, Back, Style

class ColoredFormatter(logging.Formatter):
    def __init__(self, *args, colors=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.colors = colors if colors else {}

    def format(self, record):
        record.color = self.colors.get(record.levelname, '')
        record.reset = Style.RESET_ALL
        return super().format(record)


def set_logger(level=logging.INFO):
    formatter = ColoredFormatter(
        '{color}[{levelname:.1s}] {message}{reset}',
        style='{',
        colors={
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
        }
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.handlers[:] = []
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
