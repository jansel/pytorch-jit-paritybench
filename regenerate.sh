#!/bin/bash
set -ex
python main.py 2>&1 | tee output.log
git add generated
for X in patches/*
do
    patch -p1 < $X
done
pytest generated
