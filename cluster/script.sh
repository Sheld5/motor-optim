#!/bin/sh

DATE=$(date +"%Y-%m-%d")
ID="${DATE}-${RANDOM}"
OUT="./experiments/logs/log-${ID}.txt"

# export JULIA_NUM_THREADS=12
julia ./cluster/script.jl $ID 1>$OUT 2>$OUT
