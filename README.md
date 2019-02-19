# DeepDNAseq
DNA-seq analysis with deep learning using Keras (tensorflow backend) in High performance computing (HPC) environment. DeepDNAseq makes a binary classification of the input DNA sequence after being trained with 2047 training samples.

## Installation
- Preparing HPC for running Keras. 
  - Add following modules in the bashrc file. Use "nano ~/.bashrc" for editing it, and paste following lines in that file (at the end), and save it. This will pre-load all the following modules while logging in.
``` 
    - module load cuda/9.0
    - module load cudnn/7.1.1-cuda9.0
    - module load gcc/4.9.0
    - module load python3/3.6.2
    - module load cmake/3.8.2
    - module load magma/2.3.0
    - module load intel-mkl/17.0.1.132
```
  - install keras using pip3 command. This will install keras module in your /home space in the .local/lib directory. You can use "--prefix=" if you want to install it in /short space as described in https://opus.nci.org.au/display/Help/Python.
```
      - module load tensorflow/1.8-cudnn7.1-python3.6
      - module load python3/3.6.2 
      - module load hdf5/1.10.2
      - pip3 install -v --no-binary :all: --user keras
```
  - You will need to update the PYTHONPATH variable to point to the directory you used, like this:
```
      - export PYTHONPATH=/home/561/aa7970/.local/lib/python3.6/site-packages
```
      
### Example of a bash script for running a python file
```
#!/bin/bash
#PBS -P yr31
#PBS -q gpu
#PBS -l ngpus=2
#PBS -l ncpus=6
#PBS -l walltime=0:45:00,mem=8GB
#PBS -l wd

python3 <yourPythonFile.py>
```
## Training Data
Use training dataset from a Toy training data ('toy_TrainData.csv' and 'toy_TrainLabel.csv'). This data set represents 2047 training samples (DNA sequences here) composed of random number of positive and negative samples, each having a lenth of 1000. For convenience this data set has been pre- one-hot encoded. Hence it has the dimension of [8188,1000] (2047 * 4 = 8188).

## Model
 - Training model has CNN (convolutional neural network) architecture with three convolution layer (each accompanied with a leaky Relu layer, aka activation layer; a maxpooling layer; and a dropout layer), 2 dense layer (one dropout and one leaky Relu layer in the middle).
 - input shape (1000,4,1)
 - number of classes for prediction was set to 2
 - batch size = 16
 - number of epoch = 5


## Training
 - Both training and test data sets are reshaped for matching first convolution layer which is a input layer. 
 
   - train_data = (np.arange(train_data.max()) == train_data[...,None]-1).astype('float32')
   - train_data =  train_data.reshape(2047,1000,4,1)
   - test_data = (np.arange(test_data.max()) == test_data[...,None]-1).astype('float32')
   - test_data =  test_data.reshape(500,1000,4,1)
   - history = model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size)

## Testing
 - pred_test_labels = model.predict(test_data)
