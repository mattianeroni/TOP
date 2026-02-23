from __future__ import annotations

import numpy as np 
import numpy.typing as npt

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from top.problem import Problem, Route


def get_lognormal_params(
        target_mean: npt.NDArray[np.float64], 
        target_variance: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """ In numpy lognormal function, what they call mean and sigma, looking at the source code, are actually 
    the mean and sigma of the underlying normal distribution. So we need to do the conversion. """
    sigma2 = np.log(1 + target_variance / (target_mean**2))
    sigma = np.sqrt(sigma2)
    mu = np.log(target_mean) - (sigma2 / 2)
    return mu, sigma

def get_route_stochastic_reward(p: Problem, route: Route, n_iter: int) -> float:
    """ Compute the stochastic reward of a route as (reward * p), where reward is the deterministic
    reward of the route and p is the probability to complete it in tmax. """
    dists, tmax, customers, n_legs = p.dists, p.tmax, route.customers, len(route.customers) - 1
    target_means = np.array([dists[i, j] for i, j in zip(customers[:-1], customers[1:])])
    target_vars = target_means * 0.05
    mus, sigmas = get_lognormal_params(target_means, target_vars)
    # Simulation
    rng = np.random.default_rng()
    simulated_transit_times = rng.lognormal(mus, sigmas, size=(n_iter, n_legs))
    # Check in how many of the n_iter simulations the total transit time of the route didn't exceed tmax
    feasible_route_count = (simulated_transit_times.sum(axis=1) <= tmax).astype(np.int64).sum() 
    return route.reward * (feasible_route_count / n_iter)


