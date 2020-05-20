#!/usr/bin/env bash

set -e
set -x

pytest --doctest-modules --junitxml=test-reports/junit.xml \
 --cov=kashgari --cov-report=xml:coverage.xml --cov-report=html:htmlcov --cov-config .config.ini tests
