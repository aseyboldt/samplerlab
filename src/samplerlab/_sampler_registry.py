import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, cast

from arviz import InferenceData

from samplerlab._model_registry import ModelLibrary, ModelMaker, SupportFlag, check_name


@dataclass
class TimeInfo:
    wall_time: float
    process_time: float


@dataclass
class Sampler:
    model_lib: ModelLibrary
    required_flags: set[SupportFlag]
    sample_func: Callable[[int, ModelMaker], "SampleResult"]
    keywords: list[str]
    name: str


@dataclass
class SampleResult:
    meta: dict[str, bool | int | float | str]
    trace: InferenceData
    time_info: TimeInfo
    model_maker: ModelMaker
    float32: bool
    device: Literal["cpu"] | Literal["cuda"]

    def flags(self) -> list[str]:
        result = [self.device]
        if self.float32:
            result.append("float32")
        return result


@contextmanager
def measure():
    result = {}
    wall_start = time.perf_counter()
    process_start = time.process_time()
    yield result
    wall_end = time.perf_counter()
    process_end = time.process_time()
    result["times"] = TimeInfo(
        wall_time=wall_end - wall_start, process_time=process_end - process_start
    )


_samplers: list[Sampler] = []


def _sampler_decorator(
    sample_func,
    *,
    required_flags: set[SupportFlag],
    model_lib: ModelLibrary,
    name: str | None = None,
    keywords: list[str] | None = None,
):
    if name is None:
        name = sample_func.__name__

    name = cast(str, name)
    check_name(name)

    if keywords is None:
        keywords = []

    sampler = Sampler(
        sample_func=sample_func,
        required_flags=required_flags,
        model_lib=model_lib,
        name=name,
        keywords=keywords,
    )
    _samplers.append(sampler)
    return sample_func


def stan_sampler(
    make_sampler_func=None,
    *,
    cuda=False,
    jax=False,
    numba=False,
    float32=False,
    name: str | None = None,
    keywords: list[str] | None = None,
):
    flags = set()
    if cuda:
        flags.add("cuda")
    if jax:
        flags.add("jax")
    if numba:
        flags.add("numba")
    if float32:
        flags.add("float32")

    if make_sampler_func is None:
        return partial(
            _sampler_decorator,
            model_lib="stan",
            required_flags=flags,
            name=name,
            keywords=keywords,
        )
    else:
        return _sampler_decorator(
            make_sampler_func,
            model_lib="stan",
            required_flags=flags,
            name=name,
            keywords=keywords,
        )


def pymc_sampler(
    make_sampler_func=None,
    *,
    cuda=False,
    jax=False,
    numba=False,
    float32=False,
    name: str | None = None,
    keywords: list[str] | None = None,
):
    flags = set()
    if cuda:
        flags.add("cuda")
    if jax:
        flags.add("jax")
    if numba:
        flags.add("numba")
    if float32:
        flags.add("float32")

    if make_sampler_func is None:
        return partial(
            _sampler_decorator,
            model_lib="pymc",
            required_flags=flags,
            name=name,
            keywords=keywords,
        )
    else:
        return _sampler_decorator(
            make_sampler_func,
            model_lib="pymc",
            required_flags=flags,
            name=name,
            keywords=keywords,
        )
