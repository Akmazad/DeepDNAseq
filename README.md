# DeepRNAseq
RNA-seq analysis with deep learning using PyTorch in High performance computing (HPC) environment

## Installation
- Preparing HPC for running Pytoroch. 
  - Add following modules in the bashrc file. Use "nano ~/.bashrc" for editing it, and paste following lines in that file (at the end), and save it.
  
    - module load cuda/9.0
    - module load cudnn/7.1.1-cuda9.0
    - module load gcc/4.9.0
    - module load python3/3.6.2
    - module load cmake/3.8.2
    - module load magma/2.3.0
    - module load pytorch/0.5.0a0x-py36
    - module load intel-mkl/17.0.1.132

### Example of a bash script for running a python file

#!/bin/bash
#PBS -P yr31
#PBS -q gpu
#PBS -l ngpus=2
#PBS -l ncpus=6
#PBS -l walltime=0:45:00,mem=8GB
#PBS -l wd

python3 <yourPythonFile.py>

## Training Data
Use training dataset from DeepSea

## Model

## Training

## Testing
