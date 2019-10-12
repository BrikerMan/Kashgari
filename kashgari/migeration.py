# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: migration.py
# time: 2:31 下午
import subprocess
import logging


def show_migration_guide():
    requirements = subprocess.getoutput("pip freeze")
    for package in requirements.splitlines():
        package_name, package_version = package.split('==')
        if package_name == 'kashgari-tf':
            guide = """
   ╭─────────────────────────────────────────────────────────────────────────────╮
   │                                                                             │
   │          We changed the package name for clarity and consistency.           │
   │                     From now on, it is all `kahgari`.                       │
   │    Changelog: https://github.com/BrikerMan/Kashgari/releases/tag/v1.0.0     │
   │                                                                             │
   │            | Backend          | pypi version   | desc           |           │
   │            | ---------------- | -------------- | -------------- |           │
   │            | TensorFlow 2.x   | kashgari 2.x.x | coming soon    |           │
   │            | TensorFlow 1.14+ | kashgari 1.x.x |                |           │
   │            | Keras            | kashgari 0.x.x | legacy version |           │
   │                                                                             │
   ╰─────────────────────────────────────────────────────────────────────────────╯
"""
            logging.warning(guide)


if __name__ == "__main__":
    show_migration_guide()
    print("hello, world")
