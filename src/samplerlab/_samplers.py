from contextlib import contextmanager
from functools import partial
from itertools import product
from typing import Literal

import jax
import numpy as np
import nutpie
import pymc as pm
import pytensor

from samplerlab._sampler_registry import SampleResult, measure, pymc_sampler


@contextmanager
def set_jax_config(
    device: Literal["cpu", "cuda"],
    floatX: Literal["float32", "float64"],
):
    if device == "cuda":
        dev = jax.devices("cuda")[0]
    elif device == "cpu":
        dev = jax.devices("cpu")[0]
    else:
        assert False

    old_jax_x64 = jax.config.jax_enable_x64
    old_jax_default_device = jax.config.jax_default_device
    jax.config.update("jax_enable_x64", floatX == "float64")
    jax.config.update("jax_default_device", dev)

    with jax.default_device(dev):
        yield

    jax.config.update("jax_enable_x64", old_jax_x64)
    jax.config.update("jax_default_device", old_jax_default_device)


def jax_has_cuda():
    for dev in jax.devices():
        # We should test for cuda, but I didn't find a field for that...
        if dev.platform == "gpu":
            return True
    return False


def pymc_default(seed, model_maker, floatX):
    with pytensor.config.change_flags(floatX=floatX):
        model = model_maker.make()

        with measure() as result:
            with model:
                np.random.seed(seed)
                trace = pm.sample(
                    chains=4,
                    progressbar=False,
                    discard_tuned_samples=False,
                    compute_convergence_checks=False,
                )

        time_info = result["times"]
        return SampleResult(
            meta={},
            trace=trace,
            time_info=time_info,
            model_maker=model_maker,
            float32=floatX == "float32",
            device="cpu",
        )


_floatX_types = ["float32", "float64"]
for floatX in _floatX_types:
    pymc_sampler(
        partial(pymc_default, floatX=floatX),
        float32=floatX == "float32",
        name="pymc_default",
    )


if jax_has_cuda():
    _jax_devices: list[Literal["cpu", "cuda"]] = ["cpu", "cuda"]
else:
    _jax_devices: list[Literal["cpu", "cuda"]] = ["cpu"]


def nutpie_pymc(seed, model_maker, floatX, device, backend, lowrank):
    with pytensor.config.change_flags(floatX=floatX):
        with set_jax_config(device, floatX):
            model = model_maker.make()
            compiled = nutpie.compile_pymc_model(model, backend=backend)

            with measure() as result:
                trace = nutpie.sample(
                    compiled,
                    progress_bar=False,
                    chains=4,
                    low_rank_modified_mass_matrix=lowrank,
                    seed=seed,
                )

            time_info = result["times"]

        return SampleResult(
            meta={},
            trace=trace,
            time_info=time_info,
            model_maker=model_maker,
            float32=floatX == "float32",
            device=device,
        )


_pytensor_backends: list[Literal["numba", "jax"]] = ["numba", "jax"]
for floatX, device, backend in product(_floatX_types, _jax_devices, _pytensor_backends):
    if backend == "numba" and device == "cuda":
        continue

    pymc_sampler(
        partial(
            nutpie_pymc, floatX=floatX, device=device, backend=backend, lowrank=False
        ),
        jax=backend == "jax",
        float32=floatX == "float32",
        cuda=device == "cuda",
        numba=backend == "numba",
        name=f"nutpie-{backend}",
    )
    pymc_sampler(
        partial(
            nutpie_pymc, floatX=floatX, device=device, backend=backend, lowrank=True
        ),
        jax=backend == "jax",
        float32=floatX == "float32",
        cuda=device == "cuda",
        numba=backend == "numba",
        name=f"nutpie-{backend}-lowrank",
    )


def pymc_jax_sampler(seed, model_maker, floatX, device, chain_method):
    with pytensor.config.change_flags(floatX=floatX):
        with set_jax_config(device, floatX):
            model = model_maker.make()
            with measure() as result:
                with model:
                    np.random.seed(seed)
                    trace = pm.sample(
                        chains=4,
                        nuts_sampler=backend,
                        nuts_sampler_kwargs={"chain_method": chain_method},
                        progressbar=False,
                        discard_tuned_samples=False,
                        compute_convergence_checks=False,
                    )

        time_info = result["times"]

        return SampleResult(
            meta={},
            trace=trace,
            time_info=time_info,
            model_maker=model_maker,
            float32=floatX == "float32",
            device=device,
        )


for floatX, device, backend, chain_method in product(
    _floatX_types,
    _jax_devices,
    ["numpyro", "blackjax"],
    ["parallel", "vectorized"],
):
    pymc_sampler(
        partial(
            pymc_jax_sampler, floatX=floatX, device=device, chain_method=chain_method
        ),
        jax=True,
        float32=floatX == "float32",
        cuda=device == "cuda",
        name=f"{backend}-{chain_method}",
    )
