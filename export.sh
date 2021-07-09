#!/usr/bin/env bash

./build.sh

#TODO change teamname to actual teamname

docker save teamname | gzip -c > teamname.tar.gz
