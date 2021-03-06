#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

docker run --rm --runtime nvidia --memory=10g --gpus="device=0" \
       -v ${SCRIPTPATH}/test/:/input/ -v ${SCRIPTPATH}/output/:/output/ \
       -u $UID:$GROUPS teamname
       #TODO change teamname to actual teamname
       #TODO if your method does not need a gpu, remove --gpus="device=0". If it does need a gpu, make sure you installed the nvidia container toolkit before running your docker (see: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

docker run --rm \
        -v ${SCRIPTPATH}/output/:/output/ \
        -v ${SCRIPTPATH}/test/:/input/ \
        -u $UID:$GROUPS python:3.7-slim python -c "import json, sys; f1 = json.load(open('/output/results.json')); f2 = json.load(open('/input/expected_output.json')); sys.exit(f1 != f2);"

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi
