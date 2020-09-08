#!/usr/bin/env bash

set -e

echo "delete caches"

if [ -d '_site' ] ; then
    rm -r _site
fi

if [ -d 'kashgari.egg-info' ] ; then
    rm -r kashgari.egg-info
fi

if [ -d '.coverage.imac.46767.552596' ] ; then
    rm -r .coverage.imac.46767.552596
fi

if [ -d '.pytest_cache' ] ; then
    rm -r .pytest_cache
fi

if [ -d 'tf_dir' ] ; then
    rm -r tf_dir
fi

if [ -d '_site_src' ] ; then
    rm -r _site_src
fi

if [ -d 'dist' ] ; then
    rm -r dist
fi

if [ -d 'build' ] ; then
    rm -r build
fi

if [ -d 'htmlcov' ] ; then
    rm -r htmlcov
fi

if [ -d 'test_report' ] ; then
    rm -r test_report
fi
