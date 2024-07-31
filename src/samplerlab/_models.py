from functools import partial

import numpy as np
import pandas as pd
import pymc as pm
from scipy import special

from samplerlab._model_registry import pymc_model


@pymc_model
def simple_olm():
    rng = np.random.default_rng(42)
    predictors = np.array(rng.normal(size=(10_000, 50)), order="C")
    beta = rng.normal(size=50)
    mu = predictors @ beta
    sigma = 1
    observed = mu + rng.normal(size=10_000)

    with pm.Model(check_bounds=False) as model:
        beta = pm.Normal("beta", shape=50)
        mu = predictors @ beta
        sigma = pm.HalfNormal("sigma")
        pm.Normal("y", mu=mu, sigma=sigma, observed=observed)

    return model


@pymc_model
def simple_olm_noblas():
    rng = np.random.default_rng(42)
    predictors = rng.normal(size=(10_000, 50))
    beta = rng.normal(size=50)
    mu = predictors @ beta
    sigma = 1
    observed = mu + rng.normal(size=10_000)

    with pm.Model(check_bounds=False) as model:
        beta = pm.Normal("beta", shape=50)
        mu = (predictors * beta[None, :]).sum(1)
        sigma = pm.HalfNormal("sigma")
        pm.Normal("y", mu=mu, sigma=sigma, observed=observed)

    return model


def logistic(parametrization):
    rng = np.random.default_rng(42)

    n_group1 = 100
    n_group2 = 50
    n_obs = 50_000

    intercept = rng.normal()
    group1 = rng.integers(n_group1, size=n_obs)
    group2 = rng.integers(n_group2, size=n_obs)

    group1_effect = rng.normal(size=n_group1)
    group2_effect = rng.normal(size=n_group2)
    group1_group2_effect = rng.normal(size=(n_group1, n_group2))

    mu = (
        intercept
        + group1_effect[group1]
        + group2_effect[group2]
        + group1_group2_effect[group1, group2]
    )

    noise = rng.normal(size=n_obs)

    y = rng.random(size=n_obs) < special.expit(mu + noise)

    with pm.Model(check_bounds=False) as model:
        intercept = pm.Normal("intercept")

        if parametrization == "noncentered":
            sd = pm.HalfNormal("group1_sigma")
            raw = pm.ZeroSumNormal("group1_deflect_unscaled", shape=n_group1)
            group1_deflect = pm.Deterministic("group1_deflect", sd * raw)

            sd = pm.HalfNormal("group2_sigma")
            raw = pm.ZeroSumNormal("group2_deflect_unscaled", shape=n_group2)
            group2_deflect = pm.Deterministic("group2_deflect", sd * raw)
        else:
            sd = pm.HalfNormal("group1_sigma")
            group1_deflect = pm.ZeroSumNormal(
                "group1_deflect", sigma=sd, shape=n_group1
            )

            sd = pm.HalfNormal("group2_sigma")
            group2_deflect = pm.ZeroSumNormal(
                "group2_deflect", sigma=sd, shape=n_group2
            )

        sd = pm.HalfNormal("group1_group2_sigma")
        raw = pm.ZeroSumNormal(
            "group1_group2_deflect_unscaled", shape=(n_group1, n_group2)
        )
        group1_group2_deflect = pm.Deterministic("group1_group2_deflect", sd * raw)

        mu = (
            intercept
            + group1_deflect[group1]
            + group2_deflect[group2]
            + group1_group2_deflect.ravel()[group1 * n_group2 + group2]
        )

        pm.Bernoulli("y", p=pm.math.sigmoid(mu), observed=y)

    return model


pymc_model(
    partial(logistic, parametrization="centered"),
    name="logistic-centered",
)

pymc_model(
    partial(logistic, parametrization="noncentered"),
    name="logistic-noncentered",
)


@pymc_model
def radon():
    data = pd.read_csv(pm.get_data("radon.csv"))
    data["log_radon"] = data["log_radon"].astype(np.float64)
    county_idx, counties = pd.factorize(data.county)
    coords = {"county": counties, "obs_id": np.arange(len(county_idx))}

    with pm.Model(coords=coords, check_bounds=False) as model:
        intercept = pm.Normal("intercept", sigma=10)

        raw = pm.Normal("county_raw", dims="county")
        sd = pm.HalfNormal("county_sd")
        county_effect = pm.Deterministic("county_effect", raw * sd, dims="county")

        floor_effect = pm.Normal("floor_effect", sigma=2)

        raw = pm.Normal("county_floor_raw", dims="county")
        sd = pm.HalfNormal("county_floor_sd")
        county_floor_effect = pm.Deterministic(
            "county_floor_effect", raw * sd, dims="county"
        )

        mu = (
            intercept
            + county_effect[county_idx]
            + floor_effect * data.floor.values
            + county_floor_effect[county_idx] * data.floor.values
        )

        sigma = pm.HalfNormal("sigma", sigma=1.5)
        pm.Normal(
            "log_radon",
            mu=mu,
            sigma=sigma,
            observed=data.log_radon.values,
            dims="obs_id",
        )

    return model


@pymc_model
def plain_normal():
    with pm.Model(check_bounds=False) as model:
        pm.Normal("x", shape=1000)

    return model
