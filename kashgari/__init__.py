# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz

@version: 1.0
@license: Apache Licence
@file: __init__.py.py
@time: 2019-01-19 13:42

"""
import kashgari.embeddings
import kashgari.corpus
import kashgari.tasks

from kashgari.tasks import classification
from kashgari.tasks import seq_labeling

from kashgari.macros import config

from kashgari.version import __version__
from colorama import Fore

print(f"""
   {Fore.YELLOW}╭────────────────────────────────────────────────────────────────────────────╮
   {Fore.YELLOW}│                                                                            │
   {Fore.YELLOW}│ {Fore.WHITE}                      New tf.keras version available!                      {Fore.YELLOW}│
   {Fore.YELLOW}│ {Fore.WHITE}   Changelog: {Fore.CYAN}https://github.com/BrikerMan/Kashgari/releases/tag/v0.5.0    {Fore.YELLOW}│
   {Fore.YELLOW}│ {Fore.WHITE}    Run {Fore.BLUE}`pip uninstall kashgari && pip install kashgari-tf` {Fore.WHITE}to install     {Fore.YELLOW}│
   {Fore.YELLOW}│ {Fore.WHITE}                Documents: {Fore.CYAN}https://kashgari.readthedocs.io{Fore.YELLOW}                 │
   {Fore.YELLOW}│                                                                            │ 
   {Fore.YELLOW}╰────────────────────────────────────────────────────────────────────────────╯
""")

if __name__ == "__main__":
    print("Hello world")
