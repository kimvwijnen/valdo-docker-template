#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

#TODO change findpvs to actual teamname
docker build -t findpvs "$SCRIPTPATH"
