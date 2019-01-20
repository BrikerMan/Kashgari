# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: logger.py
@time: 2019-01-19 09:58

"""
import os
import sys
import logging
from colorlog import ColoredFormatter


def init_logger():
    level = os.getenv('LOG_LEVEL', 'DEBUG')
    change_log_level(level)


def change_log_level(log_level):
    print('----------------------')
    # color_format = "%(log_color)s[%(asctime)s] %(levelname)-7s  " \
    #                "%(name)s:%(filename)s:%(lineno)d - %(message)s"
    color_format = "%(log_color)s[%(asctime)s] %(levelname)-5s " \
                   "- %(message)s"

    color_formatter = ColoredFormatter(color_format,
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
                                       style='%'
                                       )

    print_handler = logging.StreamHandler(sys.stdout)
    print_handler.setFormatter(color_formatter)
    print_handler.setLevel(log_level)

    logging.basicConfig(level=logging.DEBUG,
                        handlers=[
                            # handler,
                            print_handler
                        ])

    logging.info('logging init finished')


if __name__ == "__main__":
    init_logger()

    logging.info('info')
    logging.error('error')
    logging.warning('warning')

