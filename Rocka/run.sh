#!/usr/bin/env bash

export PYTHONPATH="./explib:./datalib:$PYTHONPATH"

exec python ./algorithm/run_cluster.py "$@"
