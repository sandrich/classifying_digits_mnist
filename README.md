![CircleCI](https://img.shields.io/circleci/build/github/sandrich/classifying_digits_mnist/master)
[![Coverage Status](https://coveralls.io/repos/github/sandrich/classifying_digits_mnist/badge.svg?branch=master)](https://coveralls.io/github/sandrich/classifying_digits_mnist?branch=master)
[![Generic badge](https://img.shields.io/badge/doc-latest-orange.svg)](https://sandrich.github.io/classifying_digits_mnist/)
![GitHub](https://img.shields.io/github/license/sandrich/classifying_digits_mnist)
![GitHub issues](https://img.shields.io/github/issues/sandrich/classifying_digits_mnist)

# M-NIST classification algorithm comparison

## Installation

You can just use
```bash
pip install mnist-classifier
```

## Documentation

You can find all the information you need on the [documentation page](https://sandrich.github.io/classifying_digits_mnist/index.html)

## Motivation for project

This project was realised in the scope of a course in Artificial Intelligence offered by [UniDistance](https://distanceuniversity.ch/artificial-intelligence/) and the [Idiap research Institute](https://github.com/idiap)

The hypothesis motivating the development of this package is the following:

 > Random Forests can give similar resulting prediction models to MLP Neural Networks on the M-NIST digit dataset in significantly less time.

With the code in this repository, we show that indeed, Random Forests *can* in fact produce similar (if not better) results with training times orders of magnitude smaller.

## License
MIT

## Authors
@sandrich - Christian Sandrini
@bigskapinsky - Calixte Mayoraz
