#!/bin/bash
set -ex
python main.py 2>&1 | tee output.log
