# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: logger.py
# time: 12:26 下午

import logging


logger = logging.getLogger(__name__)

class Logger:

    @classmethod
    def setup_logger(cls, level='DEBUG'):
        import sys
        from colorlog import ColoredFormatter
        log_format = "%(log_color)s%(asctime)s | %(levelname)-7s | %(message)s"
        color_formatter = ColoredFormatter(log_format,
                                           datefmt=None,
                                           reset=True,
                                           log_colors={
                                               'DEBUG': 'white',
                                               'INFO': 'green',
                                               'WARNING': 'purple',
                                               'ERROR': 'red',
                                               'CRITICAL': 'red,bg_white',
                                           },
                                           secondary_log_colors={},
                                           style='%')

        print_handler = logging.StreamHandler(sys.stdout)
        print_handler.setFormatter(color_formatter)
        print_handler.setLevel(level)
        logging.basicConfig(level=logging.DEBUG, format=log_format)

    @staticmethod
    def debug(message):
        from kashgari import config
        if config.verbose:
            logging.debug(message)

    @staticmethod
    def info(message):
        from kashgari import config
        if config.verbose:
            logging.info(message)

    @staticmethod
    def warning(message):
        from kashgari import config
        if config.verbose:
            logging.warning(message)


if __name__ == "__main__":
    logging.info('logging init finished')
