#!/usr/bin/env bash

set -e

docker run --rm -it -u $(id -u):$(id -g) -v  "$PWD/_site":/mnt linkchecker/linkchecker index.html
