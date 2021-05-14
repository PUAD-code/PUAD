#!/usr/bin/env bash

export PYTHONPATH="./explib:./datalib:$PYTHONPATH"

exec python ./datalib/compile.py "$@"
