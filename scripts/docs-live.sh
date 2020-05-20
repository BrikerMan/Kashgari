#!/usr/bin/env bash

echo "Build and Run API documents"

sh scripts/docs.sh

python3 -m http.server --directory _site
