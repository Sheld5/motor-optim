#!/bin/sh

for i in {1..24}
do
   sbatch -p cpulong --cpus-per-task=20 /home/soldasim/motor-optim/motor-optim/cluster/script.sh
done
