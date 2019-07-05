#! /usr/bin/env bash

for cap in 19 20 21 22 24 25 26 27 28 29 30 31 32
do
    python3 mnist.py --primary ${cap} --digit 16 --gpu 1 --epochs 100
done
