#!/usr/bin/env bash

set -e
set -x

mypy kashgari --config-file=.config.ini
flake8 kashgari --config=.config.ini
