Introduction
============

GeoTorch is a library for constrained optimisation and optimization on manifolds built on top of PyTorch. It allows for putting restrictions on the layers or arbitrary tensors in your neural network.

Motivation
**********

Although the term manifold is a bit intimidating, the applications in which one finds them are not. Examples of these are a matrix being orthogonal, invertible, or symmetric positive definite. They also appear in hyperbolic and spherical embeddings or when decomposing a matrix into its SVD or QR, or when dealing with low rank optimization, or when controlling the Lipschitz constant of a layer. Each of these examples has at least one manifold lurking behind. Even better, all these things are already implemented in GeoTorch and ready to use in one line of code.
