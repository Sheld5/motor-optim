#!/bin/sh

NAME="<experiment-name>"

DATE=$(date +"%Y-%m-%d")
ID="${DATE}-${RANDOM}"
OUT="/home/soldasim/motor-optim/motor-optim/experiments/logs-${NAME}/log-${ID}.txt"

export JULIA_NUM_THREADS=20
julia /home/soldasim/motor-optim/motor-optim/cluster/script.jl $ID $NAME 1>$OUT 2>$OUT
