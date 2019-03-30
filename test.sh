#!/usr/bin/env bash
pip install coverage
pip install nose
nosetests --cover-erase --with-coverage --cover-html --cover-html-dir=htmlcov --cover-package="kashgari" tests
coverage xml -i
~/.app/sonar-scanner/bin/sonar-scanner \
  -Dsonar.projectKey=BrikerMan_Kashgari \
  -Dsonar.organization=brikerman-github \
  -Dsonar.sources=./kashgari \
  -Dsonar.host.url=https://sonarcloud.io \
  -Dsonar.login=$SONAR_KASHGARI_TOKEN