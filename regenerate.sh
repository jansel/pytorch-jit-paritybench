#!/bin/bash
set -ex
python paritybench.py
for X in patches/*
do
  patch -p1 < $X
done
pytest generated
