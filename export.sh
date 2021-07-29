#!/usr/bin/env bash

./build.sh

#TODO change teamname to actual teamname

docker save valdotorch | gzip -c > valdotorch.tar.gz
