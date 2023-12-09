#!/bin/sh

squeue | grep soldasim | awk '{print $1}' | while read line; do
    scancel $line
done
