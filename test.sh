#!/usr/bin/env bash
pip install coverage
pip install nose
nosetests --cover-erase --with-coverage --cover-html --cover-html-dir=htmlcov --cover-package="kashgari" tests