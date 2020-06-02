Introduction
============

GeoTorch is a library for Optimization on Manifolds or Constrained Optimization built on top of PyTorch. It allows for putting restrictions on vectors, matrices, or tensors.

Motivation
**********

Although the term manifold is a bit intimidating, the applications in which one inadvertently finds them are not.  Examples of these are a matrix being orthogonal, invertible, or symmetric positive definite. One could also consider having an embedding to be living in the hyperbolic space or the sphere, or decompose a matrix into its SVD and optimize the SVD components separately, giving direct access to the singular values of the layer. Each of these examples has at least one manifold lurking behind, and it can be implemented in GeoTorch in no more than two lines of code.

Example of some maths. For example for the Lie group \\(\operatorname{SO}(n)\\) we have the map

$$
\exp \colon \mathfrak{so}(n) \to \operatorname{SO}(n)
$$
