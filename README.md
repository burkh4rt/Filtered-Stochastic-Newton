### Discriminative Bayesian Filtering Lends Momentum to the Stochastic Newton Method for Minimizing Log-Convex Functions

> To minimize the average of a set of log-convex functions, the stochastic
> Newton method iteratively updates its estimate using subsampled versions of
> the full objective's gradient and Hessian. We contextualize this optimization
> problem as sequential Bayesian inference on a latent state-space model with a
> discriminatively-specified observation process. Applying Bayesian filtering then
> yields a novel optimization algorithm that considers the entire history of
> gradients and Hessians when forming an update. We establish matrix-based
> conditions under which the effect of older observations diminishes over time,
> in a manner analogous to Polyak's heavy ball momentum. We illustrate various
> aspects of our approach with an example and review other relevant innovations
> for the stochastic Newton method.

This code reproduces the figures found in [this manuscript](https://arxiv.org/abs/2104.12949).  
It can be run on Python 3.9.1 using the included Dockerfile (see https://www.docker.com for details) and the commands:
```
docker build --no-cache -t hibiscus .
docker run --rm -ti -v $(pwd):/src hibiscus
```
