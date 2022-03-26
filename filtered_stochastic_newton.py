#!/usr/bin/env python3

"""
Discriminative Bayesian Filtering Lends Momentum
to the Stochastic Newton Method for Minimizing Log-Convex Functions

Exhibits and tests a discriminative filtering strategy for the stochastic
(batch-based) Newton method that aims to minimize the mean of log-convex
functions using sub-sampled gradients and Hessians

Runs using the provided Dockerfile (https://www.docker.com):
```
docker build --no-cache -t hibiscus .
docker run --rm -ti -v $(pwd):/home/felixity hibiscus
```
or in a virtual environment with Python3.10:
```
python3.10 -m venv turquoise
source turquoise/bin/activate
pip3 install -r requirements.txt
python3.10 filtered_stochastic_newton.py
```

"""

from __future__ import annotations

import datetime
import functools
import inspect
import logging
import os
import platform
import re
import subprocess
import sys
import time

from typing import Callable

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.optimize import minimize as sp_minimize
from numpy.linalg import *


class DiscriminativeKalmanFilter:
    """
    Implements the Discriminative Kalman Filter as described in Burkhart, M.C.,
    Brandman, D.M., Franco, B., Hochberg, L.R., & Harrison, M.T.'s "The
    discriminative Kalman filter for Bayesian filtering with nonlinear and
    nongaussian observation models." Neural Comput. 32(5), 969–1017 (2020).
    """

    def __init__(
        self,
        stateModelA: np.mat,
        stateModelGamma: np.mat,
        stateModelS: np.mat,
        posteriorMean: np.mat = None,
        posteriorCovariance: np.mat = None,
    ) -> None:
        """
        Specifies the
        state model p(hidden_t|hidden_{t-1})
            = eta_{dState}(hidden_t; stateModelA*hidden_{t-1}, stateModelGamma)
        and measurement model p(hidden_t|observed_t)
            = eta_{dState}(hidden_t; ft, Qt)
        where ft, Qt must be supplied by the user at each time step for updates
        :param stateModelA: A from eq. (2.1b)
        :param stateModelGamma: Γ from eq. (2.1b)
        :param stateModelS: S from eq. (2.1a)
        :param posteriorMean: μ_t from eq. (2.6)
        :param posteriorCovariance: Σ_t from eq. (2.6)
        """
        self.stateModelA = stateModelA
        self.stateModelGamma = stateModelGamma
        self.stateModelS = stateModelS
        self.dState = stateModelA.shape[0]
        if posteriorMean is not None:
            self.posteriorMean = posteriorMean
        else:
            self.posteriorMean = np.zeros((self.dState, 1))
        if posteriorCovariance is not None:
            self.posteriorCovariance = posteriorCovariance
        else:
            self.posteriorCovariance = self.stateModelS

    def stateUpdate(self) -> None:
        """
        Calculates the first 2 lines of eq. (2.7) in-place
        """
        self.posteriorMean = self.stateModelA * self.posteriorMean
        self.posteriorCovariance = (
            self.stateModelA * self.posteriorCovariance * self.stateModelA.T
            + self.stateModelGamma
        )

    def measurementUpdate(self, ft: np.mat, Qt: np.mat) -> None:
        """
        Given ft & Qt, calculates the last 2 lines of eq. (2.7)
        :param ft: f(x_t) from eq. (2.2)
        :param Qt: Q(x_t) from eq. (2.2)
        :return:
        """
        if not np.all(eigvals(inv(Qt) - inv(self.stateModelS)) > 1e-6):
            Qt = inv(inv(Qt) + inv(self.stateModelS))
        newPosteriorCovInv = (
            inv(self.posteriorCovariance) + inv(Qt) - inv(self.stateModelS)
        )
        self.posteriorMean = solve(
            newPosteriorCovInv,
            solve(self.posteriorCovariance, self.posteriorMean)
            + solve(Qt, ft),
        )
        self.posteriorCovariance = inv(newPosteriorCovInv)

    def predict(self, ft: np.mat, Qt: np.mat) -> tuple[np.mat, np.mat]:
        """
        Given ft & Qt, performs stateUpdate() and measurementUpdate(ft, Qt)
        :param ft: f(x_t) from eq. (2.2)
        :param Qt: Q(x_t) from eq. (2.2)
        :return: new posterior mean and covariance from applying eq. (2.7)
        """
        self.stateUpdate()
        self.measurementUpdate(ft, Qt)
        return self.posteriorMean, self.posteriorCovariance


def ArmijoStyleSearch(
    fn: Callable[[float], float],
    t0: np.mat,
    step_dir: np.mat,
    grad_fn_t0: np.mat,
) -> np.mat:
    """
    Implements a backtracking line search inspired by Armijo, L.'s
    "Minimization of functions having Lipschitz continuous first partial
    derivatives." Pacific J. Math. 16(1), 1–3 (1966).
    :param fn: callable fn for which we seek a minimum
    :param t0: starting point
    :param step_dir: direction in which to seek a minimum of fn from t0
    :param grad_fn_t0: gradient of fn at t0
    :return: reasonable step length
    """
    fn_x0 = fn(t0)
    for k in range(5):
        step_length = 2**-k
        if fn(t0 + step_length * step_dir) - fn_x0 <= float(
            0.95 * step_length * step_dir * grad_fn_t0.T
        ):
            break
    return step_length


def angular_distance(v1: np.array, v2: np.array) -> np.float:
    """
    Returns the angle in radians between two equal-length vectors v1 and v2;
    if in 2 dimensions, returns a signed angle
    :param v1: first vector
    :param v2: second vector
    :return: angle between v1 and v2 (radians)
    """
    v1_n = np.asarray(v1).ravel() / norm(v1)
    v2_n = np.asarray(v2).ravel() / norm(v2)
    if v1_n.size != 2:
        return np.arccos(np.dot(v1_n, v2_n))
    else:
        # can assign a sign when vectors are 2-dimensional
        theta1 = np.arctan2(v1_n[1], v1_n[0])
        theta2 = np.arctan2(v2_n[1], v2_n[0])
        diff_rad = theta1 - theta2
        return min(
            [diff_rad + 2 * j * np.pi for j in range(-1, 2)],
            key=lambda x: np.abs(x),
        )


def log_calls(func: Callable) -> Callable:
    """
    Logs information about a function call including inputs and output
    :param func: function to call
    :return: wrapped function with i/o logging
    """
    logger = logging.getLogger("filtered_stochastic_newton")

    @functools.wraps(func)
    def log_io(*args, **kwargs):
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        y = func(*args, **kwargs)
        logger.debug(
            f"Function {func.__name__} called with "
            + "; ".join([str(i) + ":" + str(j) for i, j in func_args.items()])
            + f" & returned: {y}"
        )
        return y

    return log_io


class LinearGaussianExample:
    """
    Example as described in section 5 of the manuscript
    """

    def __init__(self, n_total: int, d_theta: int):
        self.n_total = n_total
        self.theta = np.mat(np.random.normal(loc=1.0, size=(1, d_theta)))
        self.X = np.mat(
            np.random.multivariate_normal(
                mean=np.zeros(d_theta),
                cov=0.9 * np.eye(d_theta) + 0.1 * np.ones((d_theta, d_theta)),
                size=self.n_total,
            )
        )
        self.zeta = np.mat(np.random.normal(size=(self.n_total, 1)))
        self.Y = self.X * self.theta.T + self.zeta
        self.minimum = self.find_minimum()

        self.logger = logging.getLogger(
            os.path.basename(__file__).split(".")[0]
        )
        self.logger.debug(f"{self.theta.tolist()=}")
        self.logger.debug(f"{self.X.tolist()=}")
        self.logger.debug(f"{self.zeta.tolist()=}")
        self.logger.debug(f"{self.Y.tolist()=}")
        self.logger.debug(f"{self.minimum.tolist()=}")

    def g_i(self, theta: np.mat, i: int) -> float:
        return 1 / 2 * float(self.Y[i] - theta * self.X[i].T) ** 2

    def grad_i(self, theta: np.mat, i: int) -> np.mat:
        return theta * (self.X[i].T * self.X[i]) - self.X[i] * float(self.Y[i])

    def grad2_i(self, theta: np.mat, i: int) -> float:
        return self.grad_i(theta, i) * self.grad_i(theta, i).T

    def hess_i(self, theta: np.mat, i: int) -> float:
        return self.X[i].T * self.X[i]

    def g_idx(self, theta: np.mat, idx: list) -> float:
        g = 0
        if len(idx) == 0:
            return g
        for i in idx:
            g += self.g_i(theta, i)
        return g / len(idx)

    def grad_idx(self, theta: np.mat, idx: list) -> np.mat:
        grad = np.zeros_like(self.grad_i(self.theta, 0))
        if len(idx) == 0:
            return grad
        for i in idx:
            grad += self.grad_i(theta, i)
        return grad / len(idx)

    def grad2_idx(self, theta: np.mat, idx: list) -> float:
        grad2 = np.zeros_like(self.grad2_i(self.theta, 0))
        if len(idx) == 0:
            return float(grad2)
        for i in idx:
            grad2 += self.grad2_i(theta, i)
        return grad2 / len(idx)

    def hess_idx(self, theta: np.mat, idx: list) -> float:
        hess = np.zeros_like(self.hess_i(self.theta, 0))
        if len(idx) == 0:
            return float(hess)
        for i in idx:
            hess += self.hess_i(theta, i)
        return hess / len(idx)

    def g(self, theta: np.mat) -> float:
        return self.g_idx(theta, list(range(self.n_total)))

    def grad(self, theta: np.mat) -> np.mat:
        return self.grad_idx(theta, list(range(self.n_total)))

    def grad2(self, theta: np.mat) -> float:
        return self.grad2_idx(theta, list(range(self.n_total)))

    def hess(self, theta: np.mat) -> float:
        return self.hess_idx(theta, list(range(self.n_total)))

    @log_calls
    def random_sample(self, n_sample: np.int) -> list:
        return list(np.random.choice(range(self.n_total), n_sample))

    def find_minimum(self) -> np.mat:
        res = sp_minimize(self.g, self.theta, method="Nelder-Mead", tol=1e-6)
        return np.mat(res.x)


def run_example(
    n_total: int,
    n_steps: int,
    n_sample: int,
    d_theta: int,
    n_starts: int,
    alpha: float,
    beta: float,
) -> pd.DataFrame:
    """
    Runs a single comparison test
    :param n_total: n from the paper
    :param n_steps: number of optimization steps to perform
    :param n_sample: size of each sample (\abs{\mathcal S} from the paper)
    :param d_theta: dimension of theta
    :param n_starts: number of restarts for a given problem
    :param alpha: positive parameter for state evolution eq. (21)
    :param beta: positive parameter, <1, for state evolution eq. (21)
    :return: dataframe containing test results
    """
    eg = LinearGaussianExample(n_total=n_total, d_theta=d_theta)
    theta0 = np.mat(np.random.normal(size=(1, d_theta)))

    for run_n in range(n_starts):

        # for recording the unfiltered estimate
        theta_nf = theta0.copy()
        theta_nf_list = [theta_nf.round(3).tolist()[0]]
        g_theta_nf_list = [eg.g(theta_nf)]
        step_direction_nf_list = []
        step_direction_nf_at_theta_f_list = []

        # for recording filtered estimate;
        # initialized at same point as unfiltered
        theta_f = theta0.copy()
        theta_f_list = [theta_f.round(3).tolist()[0]]
        g_theta_f_list = [eg.g(theta_f)]
        step_direction_f_list = []

        # for recording angular comparisons
        true_step_at_theta_f_list = []
        step_angle_nf_at_theta_f_list = []
        step_angle_f_at_theta_f_list = []

        # for recording Mt's at each step
        Mt_list = [(0.0 * np.eye(d_theta)).tolist()] * 2

        stateModelA = np.mat(alpha * np.eye(d_theta))
        stateModelGamma = np.mat(beta * np.eye(d_theta))
        stateModelS = solve(
            np.eye(d_theta) - stateModelA * stateModelA.T, stateModelGamma
        )

        for j in range(n_steps):

            # draw sample to be used by all estimates
            idx = eg.random_sample(n_sample)
            g_idx = lambda t: eg.g_idx(t, idx)

            # update non-filtered estimate
            step_direction_nf = -solve(
                eg.hess_idx(theta_nf, idx) + 1e-12 * np.eye(d_theta),
                eg.grad_idx(theta_nf, idx).T,
            ).T
            grad_t0_nf = eg.grad_idx(theta_nf, idx)
            step_length_nf = ArmijoStyleSearch(
                g_idx, theta_nf, step_direction_nf, grad_t0_nf
            )
            theta_nf += step_length_nf * step_direction_nf

            # record filtered update
            theta_nf_list.append(theta_nf.round(3).tolist()[0])
            g_theta_nf_list.append(eg.g(theta_nf))
            step_direction_nf_list.append(
                step_direction_nf.round(3).tolist()[0]
            )

            # update filtered estimate
            grad_idx_f = eg.grad_idx(theta_f, idx)
            hess_idx_f = eg.hess_idx(theta_f, idx)

            # record angular comparisons
            step_direction_nf_at_theta_f = -solve(
                hess_idx_f + 1e-12 * np.eye(d_theta),
                grad_idx_f.T,
            ).T
            step_direction_nf_at_theta_f_list.append(
                step_direction_nf_at_theta_f.round(3).tolist()[0]
            )
            opt_direction_at_theta_f = eg.minimum - theta_f
            true_step_at_theta_f_list.append(
                opt_direction_at_theta_f.round(3).tolist()[0]
            )
            step_angle_nf_at_theta_f_list.append(
                angular_distance(
                    step_direction_nf_at_theta_f,
                    opt_direction_at_theta_f,
                ).round(3)
            )

            # perform filtering
            try:
                # calculate updates using the DKF
                grad_smooth, hess_smooth = DKF_f.predict(
                    grad_idx_f.T, hess_idx_f
                )

            except NameError:
                # need to instantiate DKF
                DKF_f = DiscriminativeKalmanFilter(
                    stateModelA,
                    stateModelGamma,
                    stateModelS,
                    posteriorMean=grad_idx_f.T,
                    posteriorCovariance=hess_idx_f,
                )
                grad_smooth, hess_smooth = grad_idx_f.T, hess_idx_f

            step_direction_f = -solve(
                hess_smooth + 1e-12 * np.eye(d_theta),
                grad_smooth,
            ).T
            grad_t0_f = eg.grad_idx(theta_f, idx)
            step_length_f = ArmijoStyleSearch(
                g_idx, theta_f, step_direction_f, grad_t0_f
            )
            theta_f += step_length_f * step_direction_f

            # record filtered update
            step_direction_f_list.append(step_direction_f.round(3).tolist()[0])
            theta_f_list.append(theta_f.round(3).tolist()[0])
            g_theta_f_list.append(eg.g(theta_f))

            # record angular comparisons
            step_angle_f_at_theta_f_list.append(
                angular_distance(
                    step_direction_f,
                    opt_direction_at_theta_f,
                ).round(3)
            )

            # record Mt for filtered method
            Mt_list.append(
                (
                    alpha
                    * solve(
                        alpha**2 * hess_smooth + beta * np.eye(d_theta),
                        hess_smooth,
                    )
                )
                .round(3)
                .tolist()
            )

        step_direction_nf_list.append([np.nan for _ in range(d_theta)])
        step_direction_f_list.append([np.nan for _ in range(d_theta)])
        step_angle_nf_at_theta_f_list.append(np.nan)
        step_angle_f_at_theta_f_list.append(np.nan)

        step_direction_nf_at_theta_f_list.append(
            [np.nan for _ in range(d_theta)]
        )
        true_step_at_theta_f_list.append([np.nan for _ in range(d_theta)])

        d_theta_nf_list = [np.nan] + list(
            norm(
                np.diff(
                    np.vstack([np.array(x) for x in theta_nf_list]), axis=0
                ),
                axis=1,
            )
        )
        d_from_optimum_nf_list = [
            norm(np.array(ti) - eg.minimum) for ti in theta_nf_list
        ]
        d_theta_f_list = [np.nan] + list(
            norm(
                np.diff(
                    np.vstack([np.array(x) for x in theta_f_list]), axis=0
                ),
                axis=1,
            )
        )
        d_from_optimum_f_list = [
            norm(np.array(ti) - eg.minimum) for ti in theta_f_list
        ]

        true_minimum = [
            eg.minimum.round(3).tolist()[0] for _ in range(n_steps + 1)
        ]

        run_number = [run_n for _ in range(n_steps + 1)]

        run_results = pd.DataFrame(
            {
                "step": list(range(n_steps + 1)),
                "theta_nf": theta_nf_list,
                "d_theta_nf": d_theta_nf_list,
                "d_from_optimum_nf": d_from_optimum_nf_list,
                "g_theta_nf": g_theta_nf_list,
                "step_direction_nf": step_direction_nf_list,
                "theta_f": theta_f_list,
                "d_theta_f": d_theta_f_list,
                "d_from_optimum_f": d_from_optimum_f_list,
                "g_theta_f": g_theta_f_list,
                "step_direction_f": step_direction_f_list,
                "true_minimum": true_minimum,
                "nf_step_at_theta_f": step_direction_nf_at_theta_f_list,
                "true_step_at_theta_f": true_step_at_theta_f_list,
                "step_angle_f_at_theta_f": step_angle_f_at_theta_f_list,
                "step_angle_nf_at_theta_f": step_angle_nf_at_theta_f_list,
                "Mt": Mt_list[:-1],
                "run": run_number,
            }
        )

        try:
            # add results for run to general results
            results = pd.concat([results, run_results])
        except NameError:
            # results not yet initialized
            results = run_results

        del DKF_f

    return results


def plot_sample_paths(
    results: pd.DataFrame, n_paths: int, step_list: list[int]
) -> None:
    """
    Provides len(step_list) plots, one for each max step length, comparing the
    optimization trajectories between the filtered and non-filtered methods
    :param results: dataframe containing results
    :param n_paths: number of paths to plot
    :param step_list: max length step length
    """

    for fig_n, Tmax in enumerate(step_list):

        plt.figure(fig_n)
        fig, ax = plt.subplots()

        for i in range(n_paths):
            plt.plot(
                results["theta_nf"]
                .loc[results.run == i]
                .apply(lambda x: x[0])[: Tmax + 1],
                results["theta_nf"]
                .loc[results.run == i]
                .apply(lambda x: x[1])[: Tmax + 1],
                color="#59cbe8",
                linestyle="dashed",
                label="non-filtered",
            )
            plt.plot(
                results["theta_f"]
                .loc[results.run == i]
                .apply(lambda x: x[0])[: Tmax + 1],
                results["theta_f"]
                .loc[results.run == i]
                .apply(lambda x: x[1])[: Tmax + 1],
                color="#046a38",
                linestyle="dotted",
                label="filtered",
            )

        theta_star_x, theta_star_y = results["true_minimum"].iloc[0]
        plt.plot(
            theta_star_x,
            theta_star_y,
            color="#ed1c24",
            marker="v",
            label="true min.",
        )

        theta0_x, theta0_y = results["theta_f"].iloc[0]
        plt.plot(
            theta0_x, theta0_y, color="#98a4ae", marker="o", label="init."
        )

        handles, labels = ax.get_legend_handles_labels()
        unique_labels_dict = dict(zip(labels, handles))
        if fig_n == 0:
            ax.legend(
                unique_labels_dict.values(),
                unique_labels_dict.keys(),
                fontsize="xx-large",
            )

        # plt.title(f"Optimization trajectories after {Tmax} steps")
        ax.set_ylabel("")
        ax.set_xlabel("")

        plt.savefig(f"fig{fig_n}.pdf", format="pdf")
        plt.show()


def plot_aggregate_results(results: pd.DataFrame) -> None:
    """
    Calculates and plots average results
    :param results: dataframe containing results
    """

    avg_results = results.groupby("step").mean()
    std_results = results.groupby("step").std()

    fig_n = 3
    plt.figure(fig_n)
    fig, ax = plt.subplots()

    plt.errorbar(
        avg_results.index,
        avg_results["d_from_optimum_nf"],
        yerr=2 * std_results["d_from_optimum_nf"],
        color="#59cbe8",
        linestyle="dashed",
        label="non-filtered",
    )
    plt.errorbar(
        avg_results.index,
        avg_results["d_from_optimum_f"],
        yerr=2 * std_results["d_from_optimum_f"],
        color="#046a38",
        linestyle="dotted",
        label="filtered",
    )

    ax.set_ylabel("Euclidean Distance between Estimate and Optimum")
    ax.set_xlabel("step #")

    handles, labels = ax.get_legend_handles_labels()
    unique_labels_dict = dict(zip(labels, handles))
    ax.legend(
        unique_labels_dict.values(),
        unique_labels_dict.keys(),
        fontsize="xx-large",
    )

    plt.savefig(f"fig{fig_n}.pdf", format="pdf")
    plt.show()

    fig_n = 4
    plt.figure(fig_n)
    fig, ax = plt.subplots()

    plt.errorbar(
        avg_results.index,
        avg_results["g_theta_nf"],
        yerr=2 * std_results["g_theta_nf"],
        color="#59cbe8",
        linestyle="dashed",
        label="non-filtered",
    )
    plt.errorbar(
        avg_results.index,
        avg_results["g_theta_f"],
        yerr=2 * std_results["g_theta_f"],
        color="#046a38",
        linestyle="dotted",
        label="filtered",
    )

    ax.set_ylabel(
        "Value of Total (non-sampled) Objective Function at Estimate"
    )
    ax.set_xlabel("step #")

    plt.savefig(f"fig{fig_n}.pdf", format="pdf")
    plt.show()

    fig_n = 5
    plt.figure(fig_n)
    fig, ax = plt.subplots()

    plt.errorbar(
        avg_results.index,
        avg_results["d_theta_nf"],
        yerr=2 * std_results["d_theta_nf"],
        color="#59cbe8",
        linestyle="dashed",
        label="non-filtered",
    )
    plt.errorbar(
        avg_results.index,
        avg_results["d_theta_f"],
        yerr=2 * std_results["d_theta_f"],
        color="#046a38",
        linestyle="dotted",
        label="filtered",
    )

    ax.set_ylabel("Euclidean Norm of Step from Previous Estimate")
    ax.set_xlabel("step #")

    plt.savefig(f"fig{fig_n}.pdf", format="pdf")
    plt.show()


def generate_aggregate_angular_results_2d(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregates angular results
    :param results: dataframe containing results
    :return: dataframe comparing angle from the optimum
        for filtered & nonfiltered steps
    """

    mse_f_step_angle = (
        results.set_index("step")
        .step_angle_f_at_theta_f.apply(lambda x: x**2)
        .groupby("step")
        .mean()
    )
    mse_nf_step_angle = (
        results.set_index("step")
        .step_angle_nf_at_theta_f.apply(lambda x: x**2)
        .groupby("step")
        .mean()
    )

    bias_f_step_angle = results.groupby("step")[
        "step_angle_f_at_theta_f"
    ].mean()
    bias_nf_step_angle = results.groupby("step")[
        "step_angle_nf_at_theta_f"
    ].mean()

    err_var_f_step_angle = results.groupby("step")[
        "step_angle_f_at_theta_f"
    ].var()
    err_var_nf_step_angle = results.groupby("step")[
        "step_angle_nf_at_theta_f"
    ].var()

    angular_aggregate_results = pd.DataFrame(
        {
            "mse_f_step_angle": mse_f_step_angle,
            "mse_nf_step_angle": mse_nf_step_angle,
            "bias_f_step_angle": bias_f_step_angle,
            "bias_nf_step_angle": bias_nf_step_angle,
            "err_var_f_step_angle": err_var_f_step_angle,
            "err_var_nf_step_angle": err_var_nf_step_angle,
        }
    )

    return angular_aggregate_results


def generate_aggregate_Mt_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregates results for the momentum term Mt
    :param results: dataframe containing results
    :return: dataframe comparing angle from the optimum
        for filtered & nonfiltered steps
    """

    grouped_Mt = results.groupby("step").Mt

    avg_Mt = [
        y.apply(np.array).agg(np.mean).round(3).tolist() for _, y in grouped_Mt
    ]
    std_Mt = [
        y.apply(np.array).to_numpy().std().round(3).tolist()
        for _, y in grouped_Mt
    ]
    avg_rho_Mt = [
        y.apply(lambda x: np.max(eig(np.array(x))[0]))
        .to_numpy()
        .mean()
        .round(3)
        .tolist()
        for _, y in grouped_Mt
    ]
    std_rho_Mt = [
        y.apply(lambda x: np.max(eig(np.array(x))[0]))
        .to_numpy()
        .std()
        .round(3)
        .tolist()
        for _, y in grouped_Mt
    ]
    max_rho_Mt = [
        y.apply(lambda x: np.max(eig(np.array(x))[0]))
        .to_numpy()
        .max()
        .round(3)
        .tolist()
        for _, y in grouped_Mt
    ]

    aggregate_Mt_results = pd.DataFrame(
        {
            "step": list(range(len(avg_Mt))),
            "avg_Mt_coordinatewise": avg_Mt,
            "std_Mt_coordinatewise": std_Mt,
            "avg_rho_Mt": avg_rho_Mt,
            "std_rho_Mt": std_rho_Mt,
            "max_rho_Mt": max_rho_Mt,
        }
    ).drop(
        0
    )  # Mt isn't defined for the first step

    return aggregate_Mt_results


def start_logger() -> logging.Logger:
    """
    Keeps records
    :return: logger that writes to both screen and log file
    """
    logger = logging.getLogger("filtered_stochastic_newton")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%dT%H:%M:%S%z"
    )

    now = re.sub(
        "[-:]", "", datetime.datetime.now().replace(microsecond=0).isoformat()
    )

    # write logs to file
    fh = logging.FileHandler(
        "filtered_stochastic_newton_" + now + ".log", mode="w"
    )
    fh.setLevel(logging.DEBUG)
    # fh.setLevel(logging.WARNING)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # print logs to screen
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(
        "Running Python "
        + re.sub("\n", "", sys.version)
        + " on "
        + platform.platform()
        + "."
    )
    logger.debug("Platform u-name: " + repr(platform.uname()))

    logger.info("Logging environment...")
    logger.debug(
        subprocess.check_output(
            "$(which python) -m pip list | tail -n +3",
            shell=True,
            stderr=subprocess.STDOUT,
        )
    )

    logger.info("Logging file...")
    with open(__file__, "r") as f:
        logger.debug(f.read().encode("utf-8"))

    return logger


def set_display_options() -> None:
    """
    Sets pandas and matplotlib display options
    """
    pd.options.display.width = 250
    pd.options.display.max_columns = 10
    pd.options.display.max_columns = 30
    pd.options.display.max_colwidth = 100

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10"],
        }
    )


def main() -> None:
    # initialize logger
    logger = start_logger()

    # encourage reproducibility
    np.random.seed(0)

    # run experiment
    results = run_example(
        n_total=100,
        n_steps=30,
        n_sample=5,
        d_theta=2,
        n_starts=1000,
        alpha=0.9,
        beta=0.2,
    )
    # logger.critical(results.to_markdown().encode("utf-8"))

    set_display_options()
    print(results)

    # generate plots for 3 sample optimization trajectories
    plot_sample_paths(
        results=results, n_paths=3, step_list=list(range(10, 40, 10))
    )

    # generate aggregate result plots
    plot_aggregate_results(results=results)

    # generate and display aggregate angular results
    aggregate_angular_results = generate_aggregate_angular_results_2d(results)
    logger.critical(aggregate_angular_results.to_markdown().encode("utf-8"))
    print(
        aggregate_angular_results.loc[:4][
            [
                "mse_nf_step_angle",
                "mse_f_step_angle",
            ]
        ]
        .T.round(3)
        .to_latex()
    )

    # generate and display aggregate Mt results
    aggregate_Mt_results = generate_aggregate_Mt_results(results)
    logger.critical(aggregate_Mt_results.to_markdown().encode("utf-8"))
    print(aggregate_Mt_results)


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"main() executed in {time.time() - start:.2f} seconds.")
