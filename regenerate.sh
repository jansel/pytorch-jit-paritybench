#!/bin/bash
set -ex
python main.py 2>&1 | tee output.log
for X in patches/*
do
    patch -p1 < $X
done
pytest generated
