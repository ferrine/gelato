# Gelato
[![](https://travis-ci.org/ferrine/gelato.svg?branch=master)](https://travis-ci.org/ferrine/gelato)
[![Coverage Status](https://coveralls.io/repos/github/ferrine/gelato/badge.svg?branch=master)](https://coveralls.io/github/ferrine/gelato?branch=master)
## Bayesian dessert for Lasagne

![](img/gelato.jpg)

# About
Recent results in Bayesian statistics for constructing robust neural networks have proved that it is one of the best ways to deal with uncertainty, overfitting but still having good performance. Gelato will help to use bayes for neural networks.
Library heavily relies on [Theano](https://github.com/Theano/Theano), [Lasagne](https://github.com/Lasagne/Lasagne) and [PyMC3](https://github.com/pymc-devs/pymc3).

Installation
------------

* from github (assumes bleeding edge pymc3 installed)
    ```bash
    # pip install git+git://github.com/pymc-devs/pymc3.git
    pip install git+https://github.com/ferrine/gelato.git
    ```
* from source
    ```bash
    git clone https://github.com/ferrine/gelato
    pip install -r gelato/requirements.txt
    pip install -e gelato
    ```

Usage
-----
I use generic approach for decorating all Lasagne at once. Thus, for using Gelato you need to replace import statements for layers only. For constructing a network you need to be the in pm.Model context environment.

**Warning**
 - `lasagne.layers.noise`, `lasagne.layers.normalization` are not supported yet

Examples
--------
For comprehensive example of using `Gelato` you can reference [this](https://github.com/ferrine/gelato/blob/master/examples/mnist.ipynb) notebook 

Life Hack
---------
Any `spec` class can be used standalone so feel free to use it everywhere

References
----------
Charles Blundell et al: "Weight Uncertainty in Neural Networks" ([arXiv preprint arXiv:1505.05424](https://arxiv.org/abs/1505.05424))
