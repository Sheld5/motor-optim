#!/bin/sh
# Run from ~/home/soldasim/motor-optim/motor-optim
for i in {1..20}
do
   sbatch -p cpulong ./cluster/script.sh
done
