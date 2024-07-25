import warnings
import trace_events
from datetime import datetime
import watermark
import subprocess
import jax
import pytensor
import nutpie
import pandas as pd
import numpy as np
import pymc as pm
import threadpoolctl
from scipy import special

_has_cuda = False


def _setup_trace():
    watermark_info = watermark.watermark(
        conda=True,
        gpu=True,
        python=True,
        packages="numpy,scipy,nutpie,numpyro,blackjax,jax,torch,pymc,pytensor",
        machine=True,
    )
    try:
        output_smi = subprocess.check_output(["nvidia-smi"]).decode()
    except subprocess.CalledProcessError:
        output_smi = "<failed>"
    numpy_info = np.__config__.show(mode="dicts")
    jax_info = jax.print_environment_info(True)

    trace_events.init_trace(
        trace_file_dir=f'traces/{datetime.now().isoformat(timespec="seconds")}',
        trace_file_name="trace.json",
        save_at_exit=True,
        overwrite_trace_files=False,
    )

    prof = trace_events.profiler.global_profiler()
    prof._trace.add_data(
        {
            "numpy": numpy_info,
            "watermark": watermark_info,
            "nvidia-smi": output_smi,
            "jax": jax_info,
        }
    )
    warnings.filterwarnings(
        action="ignore", message="Explicitly requested dtype float64 requested"
    )
    warnings.filterwarnings(
        action="ignore", message="is incompatible with multithreaded code"
    )

    def check_gpu():
        # Get a list of devices available to JAX
        devices = jax.devices()

        # Check if any of the devices are GPUs
        for device in devices:
            if "gpu" in device.platform.lower():
                return True

        return False

    global _has_cuda
    _has_cuda = check_gpu()


def _run_model(model_func, name, floatX, sample_args):
    with pytensor.config.change_flags(floatX=floatX):
        jax.config.update("jax_enable_x64", floatX == "float64")
        settings = {"floatX": floatX, "model": name, "device": "cpu"}

        model = model_func()

        with trace_events.timeit(
            "sample",
            args={
                **settings,
                "sampler": "default",
            },
        ):
            with model:
                pm.sample(progressbar=False, **sample_args)
        with trace_events.timeit(
            "compile", args={**settings, "sampler": "nutpie-default"}
        ):
            compiled = nutpie.compile_pymc_model(model)
        with trace_events.timeit(
            "sample", args={**settings, "sampler": "nutpie-default"}
        ):
            nutpie.sample(compiled, progress_bar=False, **sample_args)

        if _has_cuda:
            devices = ["cpu", "cuda"]
        else:
            devices = ["cpu"]

        for device in devices:
            jax.config.update("jax_default_device", jax.devices(device)[0])
            settings = {**settings, "device": device}

            with trace_events.timeit("make_model", args={"model": name}):
                model = model_func()
            with trace_events.timeit("sample", args={**settings, "sampler": "numpyro"}):
                with model:
                    pm.sample(
                        nuts_sampler="numpyro",
                        progressbar=False,
                        **sample_args,
                    )
            with trace_events.timeit(
                "sample", args={**settings, "sampler": "numpyro-vectorized"}
            ):
                with model:
                    pm.sample(
                        nuts_sampler="numpyro",
                        progressbar=False,
                        nuts_sampler_kwargs={"chain_method": "vectorized"},
                        **sample_args,
                    )
            with trace_events.timeit(
                "sample",
                args={**settings, "sampler": "blackjax-vectorized"},
            ):
                with model:
                    pm.sample(
                        nuts_sampler="blackjax",
                        nuts_sampler_kwargs={"chain_method": "vectorized"},
                        progressbar=False,
                        **sample_args,
                    )
            with trace_events.timeit(
                "compile", args={**settings, "sampler": "nutpie-jax"}
            ):
                compiled = nutpie.compile_pymc_model(model, backend="jax")
            with trace_events.timeit(
                "sample", args={**settings, "sampler": "nutpie-jax"}
            ):
                nutpie.sample(compiled, progress_bar=False, **sample_args)


_benchmarks = {}


def benchmark(model_func):
    name = model_func.__name__
    _benchmarks[name] = model_func
    return model_func


def _run_benchmarks():
    for name, model_func in _benchmarks.items():
        sample_args = {"chains": 4}

        with trace_events.timeit("benchmark_model", args={"model": name}):
            for floatX in ["float64"]:
                _run_model(model_func, name, floatX, sample_args)


@benchmark
def simple_olm():
    rng = np.random.default_rng(42)
    predictors = rng.normal(size=(10_000, 50))
    beta = rng.normal(size=50)
    mu = predictors @ beta
    sigma = 1
    observed = mu + rng.normal(size=10_000)

    with pm.Model() as model:
        beta = pm.Normal("beta", shape=50)
        mu = predictors @ beta
        sigma = pm.HalfNormal("sigma")
        pm.Normal("y", mu=mu, sigma=sigma, observed=observed)

    return model


@benchmark
def logistic():
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

    with pm.Model() as model:
        intercept = pm.Normal("intercept")

        sd = pm.HalfNormal("group1_sigma")
        raw = pm.ZeroSumNormal("group1_deflect_unscaled", shape=n_group1)
        group1_deflect = pm.Deterministic("group1_deflect", sd * raw)

        sd = pm.HalfNormal("group2_sigma")
        raw = pm.ZeroSumNormal("group2_deflect_unscaled", shape=n_group2)
        group2_deflect = pm.Deterministic("group2_deflect", sd * raw)

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


@benchmark
def radon():
    data = pd.read_csv(pm.get_data("radon.csv"))
    data["log_radon"] = data["log_radon"].astype(np.float64)
    county_idx, counties = pd.factorize(data.county)
    coords = {"county": counties, "obs_id": np.arange(len(county_idx))}

    with pm.Model(coords=coords, check_bounds=True) as model:
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


@benchmark
def plain_normal():
    with pm.Model() as model:
        pm.Normal("x", shape=1000)

    return model


def main():
    _setup_trace()
    _run_benchmarks()


if __name__ == "__main__":
    main()
