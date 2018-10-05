#!/bin/bash
#PBS -P yr31
#PBS -q gpu
#PBS -l ngpus=2
#PBS -l ncpus=6
#PBS -l walltime=01:45:00,mem=64GB
#PBS -l wd

python3 toyDL2.py