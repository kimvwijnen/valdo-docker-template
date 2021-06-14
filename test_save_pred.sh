#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

mkdir -p ${SCRIPTPATH}/output/images/
# docker is run with user "algorithm" (see DockerFile), so need to set folder permissions
# to allow any user to write to the output folder
chmod 777 -R ${SCRIPTPATH}/output/

docker run --rm --runtime nvidia --memory=10g --gpus="device=0" \
       -v ${SCRIPTPATH}/test/:/input/ -v ${SCRIPTPATH}/output/:/output/ \
       findpvs

docker run --rm \
        -v ${SCRIPTPATH}/output/:/output/ \
        -v ${SCRIPTPATH}/test/:/input/ \
        python:3.7-slim python -c "import json, sys; f1 = json.load(open('/output/results.json')); f2 = json.load(open('/input/expected_output.json')); sys.exit(f1 != f2);"

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi