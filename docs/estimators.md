# Estimators

This document summarizes the estimator code used in the project. The goal is not to provide a full API reference, but to explain the main design choices and how the estimator classes relate to the methodology used in the Master's thesis.

## Overview

The estimator code is organized around robust image averaging. Given a set of aligned cryo-EM particle images, the estimators produce a representative average while trying to reduce the influence of noisy images, outliers, or images that are less compatible with the current reference.

The central idea is to assign weights to the input images, either locally at each pixel/frequency or globally per image, and then compute a weighted average. Different estimators differ mainly in how these weights are computed and in which representation of the images they operate.

## Core data structures

### `ImageBatch`

`ImageBatch` is the canonical input container for the estimators. It stores the image stack and provides access to both real-space and Fourier-space representations.

It can be constructed from real-space images:

```python
batch = ImageBatch.from_real(images)
```

or from Fourier-space images:

```python
batch = ImageBatch.from_fourier(fourier_images, real_shape=(h, w))
```

The class computes missing representations lazily. For example, if only real-space images are provided, the Fourier representation is computed when needed. It also stores or estimates variance information, which is used by estimators that normalize residuals by a pixel- or frequency-dependent scale.

### `EstimatorResult`

All estimators return an `EstimatorResult`. This object contains:

- `average`: the reconstructed real-space average, when available;
- `estimate`: the estimate in the estimator's native space;
- `weights`: the final weights used by the estimator;
- `converged`: whether the algorithm met its convergence criterion;
- `n_iter`: the number of iterations performed.

This makes the output format consistent across estimators that work in real space, Fourier space, or through combined strategies.

### `WeightSet`

`WeightSet` stores weights associated with different image spaces:

- real-space weights;
- Fourier real-part weights;
- Fourier imaginary-part weights.

This is useful because some estimators produce only real-space weights, while Fourier estimators may produce separate weights for the real and imaginary parts or shared weights for a complex Fourier representation.

## IRLS estimators

### `IRLSSolver`

`IRLSSolver` implements an Iteratively Reweighted Least Squares procedure for a single image representation.

At each iteration:

1. A weight function is evaluated using the current reference.
2. A weighted least-squares update is computed.
3. The reference is updated.
4. Convergence is checked using the relative change between consecutive references.

When all weights are equal to one, the estimator reduces essentially to the ordinary sample mean. Robust behavior comes from weight functions that downweight images, pixels, or frequencies with large residuals or low similarity to the reference.

The solver can operate in several spaces depending on the `space` argument:

- `ImageSpace.REAL`;
- `ImageSpace.FOURIER_REAL`;
- `ImageSpace.FOURIER_IMAG`;
- `ImageSpace.FOURIER_COMPLEX`.

The same solver class is therefore reused for real-space and Fourier-space estimation.

## Fourier-space estimators

The project contains several Fourier-space variants. They share the goal of estimating a real-space average through operations performed in the Fourier domain, but differ in how they treat complex Fourier coefficients.

### `IRLSFourier`

`IRLSFourier` applies two separate IRLS solvers:

- one to the real part of the Fourier coefficients;
- one to the imaginary part of the Fourier coefficients.

The two estimates are then combined into a complex Fourier estimate and transformed back to real space.

This strategy is simple and makes it possible to use different variance estimates for the real and imaginary components.

### `JointIRLSFourier`

`JointIRLSFourier` applies a single IRLS solver directly to the complex Fourier representation.

Instead of treating real and imaginary parts as independent scalar images, it works with complex residuals and uses a shared weighting scheme based on the modulus of the complex residual. This is useful when the weight should depend on the joint behavior of the complex coefficient rather than on its two components separately.

### `FlatteningIRLSFourier`

`FlatteningIRLSFourier` converts the complex Fourier representation into a real-valued flattened representation where real and imaginary parts are stored together. It then applies a standard IRLS solver to this flattened representation.

This provides another way of applying real-valued IRLS machinery to complex Fourier data, which is necessary for some global weighting schemes such as Huang and Tagare's weights.

## Recursive GMM estimator

`RecursiveGMMEstimator` uses a distance function to compare each image with the current reference. Here, "distance" is understood in the loose computational sense described in the section on distance functions; it does not necessarily refer to a formal mathematical metric. The resulting distances are modeled with a two-component Gaussian mixture model.

At each iteration:

1. Distances to the current reference are computed.
2. A two-component Gaussian mixture is fitted to the distance distribution.
3. The component with lower mean distance is interpreted as the inlier component.
4. Posterior probabilities for that component are used as weights.
5. A new weighted average is computed.

This provides a classification-like robust averaging method, where images are softly assigned to an inlier or outlier group.

*Note: the term 'distance' is used loosely here to mean any function that intends to measure dissimilarity between two images. It does not necessarily mean the function is a proper metric which is symmetric, positive non-degenerate and satisfies the triangle inequality; or even a dissimilarity metric in the traditional sense.*

## ADMM estimator

`ADMMSolver` combines real-space and Fourier-space estimation through an Alternating Direction Method of Multipliers scheme.

The method alternates between:

- a real-space IRLS update;
- a Fourier-space IRLS update;
- an update of the dual variables enforcing consistency between the two representations.

The final estimate combines the real-space and Fourier-space reconstructions.

This estimator is useful for testing whether real-space and Fourier-space robust estimates can be coupled in a single optimization scheme to improve the estimation results.

## Weight functions

Weight functions define the robust behavior of the IRLS estimators. The project includes several options, such as:

- Huber weights
- Smooth redescending weights
- Cauchy weights
- Student-t weights
- Tagare-type global structural weights
- Cosine-similarity and correlation-based weights.

Pixelwise weights produce tensors with the same spatial shape as the image stack. Global structural weights produce one scalar weight per image, broadcastable over the spatial dimensions.

## Distance functions

Distance functions are mainly used by the GMM estimator. They quantify how far each image is from a reference image. Implemented distances include:

- L1 distance
- L2 distance
- Lp distance
- Distances derived from similarity measures such as Tagare weights, cosine similarity, or cross-correlation.

### Note on the term "distance"

In this project, the term *distance* is used in a loose computational sense: it refers to any function intended to quantify how dissimilar an image is from a reference image.

These functions are not necessarily mathematical metrics. In particular, they may fail to be symmetric, positive non-degenerate, or to satisfy the triangle inequality. Some of them are transformations of similarity scores rather than distances in the strict sense.

The term is therefore used as a practical name for the quantities passed to the GMM-based estimator, not as a formal metric-space definition.

## Design philosophy

The estimator code is intended to support experimentation rather than provide a polished general-purpose cryo-EM software package. The main design goals are:

- Keep estimator inputs and outputs consistent
- Support both real-space and Fourier-space representations
- Make weight functions interchangeable
- Make the evaluation pipeline independent from the details of each estimator
- Keep enough structure for the code to be understandable and reproducible.
