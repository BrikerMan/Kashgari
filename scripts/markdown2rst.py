# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: markdown2rst.py
# time: 12:08 上午

import sys
from m2r import convert


def convert_file(file_path: str, target_path: str = None):
    if target_path is None:
        target_path = file_path.replace('.md', '.rst')

    with open(file_path, 'r') as f:
        md_content = f.read()

    with open(target_path, 'w') as f:
        f.write(convert(md_content))
        print(f'Saved RST file to {target_path}')


if __name__ == "__main__":
    convert_file(*sys.argv[1:])
