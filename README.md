# Autodiff

_Static reverse-mode automatic differentiation in C_

This repository consists of a [scalar-valued reverse-mode automatic differentiation library](lib/autodiff.c), extended into a [tensor computation library](lib/tensor.c), used as the foundation of a [multilayer perceptron model](mlp-gen.c) that scores [96% accuracy on the MNIST database](mlp-fit.c). Also included is a [curve fitting demo](curve-fit.c).

The multilayer perceptron works in two stages: in [the first](mlp-gen.c) it builds a computation graph for the model then generates C source code that directly computes the gradient of the cost function with respect to model parameters, and in [the second](mlp-fit.c) it compiles that C source code as a library and uses it for gradient descent. The [curve fitting demo](curve-fit.c), on the other hand, builds a computation graph then runs an interpreter over it in a single stroke.

Run the multilayer perceptron against MNIST with:

```sh
make bin/mlp-fit && bin/mlp-fit
```

Run the curve fitting demo with:

```sh
make bin/curve-fit && bin/curve-fit
```
