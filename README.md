# Gelato
[![](https://travis-ci.org/ferrine/gelato.svg?branch=master)](https://travis-ci.org/ferrine/gelato)
## Bayesian dessert for Lasagne

![](img/gelato.jpg)

# About
Recent results in bayesian statistics for constructing robust neural networks have proved that it is one of the best ways to deal with uncertainty, overfitting but still having good performance. Gelato will help to use bayes for neural networks.
Library heavily relies on [Theano](https://github.com/Theano/Theano), [Lasagne](https://github.com/Lasagne/Lasagne) and [PyMC3](https://github.com/pymc-devs/pymc3).

Installation
------------

```bash
git clone https://github.com/ferrine/gelato
cd gelato
pip install -r requirements.txt
pip install .
```

Usage
-----
I use generic approach for decorating all Lasagne at once. Thus, for using Gelato you need to replace import statements for layers only. For constructing a network you need to be the in pm.Model context environment.

**Warning**
 - `gelato.layers.helper` module, it is not equivalent to `lasagne.layers.helper`, it declares only `get_output` function.
 - `lasagne.layers.noise`, `lasagne.layers.normalization` are not supported yet


```python
# TODO
# if you don't pass `vp` to `get_output` you will get output without replacements in graph
```

Life Hack
---------
Any `spec` class can be used standalone so feel free to use it everywhere (e.g. in [keras](https://github.com/fchollet/keras))

References
----------
Charles Blundell et al: "Weight Uncertainty in Neural Networks" ([arXiv preprint arXiv:1505.05424](https://arxiv.org/abs/1505.05424))
