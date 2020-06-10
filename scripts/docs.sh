#!/usr/bin/env bash

set -e

echo "Build API documents to _site folder"

if [ -d '_site_src' ] ; then
    rm -r _site_src
fi

cp -r docs _site_src

if [ -d 'site' ] ; then
    rm -r site
fi

sphinx-build _site_src _site -n -a -T
