#!/usr/bin/env bash

./build.sh

docker save findpvs | gzip -c > FindPVS.tar.gz
