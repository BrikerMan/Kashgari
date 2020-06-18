#!/usr/bin/env bash

echo "Build and Run API documents"

sh scripts/docs-generate.sh

python3 -m http.server --directory _site
