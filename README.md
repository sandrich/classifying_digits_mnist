![CircleCI](https://img.shields.io/circleci/build/github/sandrich/classifying_digits_mnist/master)
[![Coverage Status](https://coveralls.io/repos/github/sandrich/classifying_digits_mnist/badge.svg?branch=master)](https://coveralls.io/github/sandrich/classifying_digits_mnist?branch=master)
[![Generic badge](https://img.shields.io/badge/doc-latest-orange.svg)](https://sandrich.github.io/classifying_digits_mnist/)
![GitHub](https://img.shields.io/github/license/sandrich/classifying_digits_mnist)
![GitHub issues](https://img.shields.io/github/issues/sandrich/classifying_digits_mnist)

# Classifying digits using 28x28px images in one of 10 classes
Small classifier for 28x28px handwritten digits based on [M-NIST](http://yann.lecun.com/exdb/mnist/) dataset 

## License
MIT

## Requirements
For this project to run properly you will need:

 - conda ([installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/))

## Installation
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

## Usage

The program can run without parameters which will take our researched value. Feel free to use different parameters to play with the data and algorithm

```bash
$ python mnist_predict.py -h
usage: mnist_predict.py [-h] [--trees TREES] [--depth DEPTH] [--impurity_method {entropy,gini}]

Run MNIST classifier

optional arguments:
  -h, --help            show this help message and exit
  --trees TREES         Number of trees
  --depth DEPTH         Maximum tree depth
  --impurity_method {entropy,gini}
                        Impurity method
```

### Example

```bash
# python mnist_predict.py 
No local fit dataset found.
Downloading fit data
['================================================='>'']]
Downloading fit labels
['================================================='>'']
No local test dataset found.
Downloading test data
['================================================='>'']]
Downloading test labels
['================================================='>'']
Starting training...
Done training.
Predicting...
Predicting...
Classification stats:
-----------------
Max tree depth: 9
Number of trees: 20
Impurity method: entropy
-----------------
Train Accuracy: 0.946
Train Accuracy: 0.935
```

## Authors
@sandrich - Christian Sandrini
@bigskapinsky - Calixte Mayoraz
