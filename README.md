### Discriminative Bayesian Filtering Lends Momentum to the Stochastic Newton Method for Minimizing Log-Convex Functions

> To minimize the average of a set of log-convex functions, the stochastic
> Newton method iteratively updates its estimate using subsampled versions of
> the full objective's gradient and Hessian. We contextualize this optimization
> problem as sequential Bayesian inference on a latent state-space model with a
> discriminatively-specified observation process. Applying Bayesian filtering
> then yields a novel optimization algorithm that considers the entire history
> of gradients and Hessians when forming an update. We establish matrix-based
> conditions under which the effect of older observations diminishes over time,
> in a manner analogous to Polyak's heavy ball momentum. We illustrate various
> aspects of our approach with an example and review other relevant innovations
> for the stochastic Newton method.

This code reproduces the figures found in
[this manuscript](https://arxiv.org/abs/2104.12949). It can be run on Python
3.10.2 using the provided Dockerfile (https://www.docker.com):

```
docker build -t hibiscus .
docker run --rm -ti -v $(pwd):/home/felixity hibiscus
```

or in a virtual environment with Python3.10 (https://docs.python.org/3.10/):

```
python3.10 -m venv turquoise
source turquoise/bin/activate
pip3 install -r requirements.txt
python3.10 filtered_stochastic_newton.py
```

<!---
Format code with:
```sh
black .
prettier --write --print-width 79 --prose-wrap always **/*.md
```
--->
