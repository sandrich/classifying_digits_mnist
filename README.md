[![sandrich](https://circleci.com/gh/sandrich/classifying_digits_mnist.svg?style=svg)](https://circleci.com/gh/sandrich/classifying_digits_mnist)

# Classifying digits using 28x28px images in one of 10 classes
Small classifier for 28x28px handwritten digits based on [M-NIST](http://yann.lecun.com/exdb/mnist/) dataset 

## License
MIT

# Requirements
For this project to run properly you will need:

 - conda ([installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/))

# Installation
To use and reproduce this project, first clone this repository in the directory of your choice
```shell script
cd /path/to/your/directory
git clone https://github.com/sandrich/classifying_digits_mnist.git
``` 

Then create a conda environment with the correct dependencies:
```shell script
conda env create --file environment.yml
```
Once the conda has finished installing all the dependencies, activate it:
```shell script
conda activate mnist_classifier
```


## Authors
@sandrich - Christian Sandrini
@bigskapinsky - Calixte Mayoraz
