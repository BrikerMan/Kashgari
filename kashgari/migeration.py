# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: migration.py
# time: 2:31 下午
import subprocess
import logging


guide = """
╭─────────────────────────────────────────────────────────────────────────╮
│ ◎ ○ ○ ░░░░░░░░░░░░░░░░░░░░░  Important Message  ░░░░░░░░░░░░░░░░░░░░░░░░│
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│              We renamed again for consistency and clarity.              │
│                   From now on, it is all `kashgari`.                    │
│  Changelog: https://github.com/BrikerMan/Kashgari/releases/tag/v1.0.0   │
│                                                                         │
│         | Backend          | pypi version   | desc           |          │
│         | ---------------- | -------------- | -------------- |          │
│         | TensorFlow 2.x   | kashgari 2.x.x | coming soon    |          │
│         | TensorFlow 1.14+ | kashgari 1.x.x |                |          │
│         | Keras            | kashgari 0.x.x | legacy version |          │
│                                                                         │
╰─────────────────────────────────────────────────────────────────────────╯
"""


def show_migration_guide():
    requirements = subprocess.getoutput("pip freeze")
    for package in requirements.splitlines():
        if '==' in package:
            package_name, package_version = package.split('==')
            if package_name == 'kashgari-tf':
                logging.warning(guide)


if __name__ == "__main__":
    show_migration_guide()
    print("hello, world")
