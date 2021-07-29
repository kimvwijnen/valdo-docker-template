#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

#TODO change teamname to actual teamname
docker build -t valdotorch "$SCRIPTPATH"
