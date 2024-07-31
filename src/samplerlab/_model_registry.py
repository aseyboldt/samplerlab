import re
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal, cast

ModelLibrary = Literal["pymc", "stan"]
SupportFlag = Literal["jax", "numba", "cuda", "float32"]


def check_name(name: str) -> None:
    pattern = r"^[a-zA-Z0-9_-]+$"
    if re.match(pattern, name) is None:
        raise ValueError(f"Invalid name for model or sampler: {name}")


@dataclass
class ModelMaker:
    name: str
    model_library: ModelLibrary
    supported_flags: set[SupportFlag]
    make_model: Callable[[], Any]
    keywords: list["str"]

    def make(self):
        return self.make_model()


_models: list[ModelMaker] = []


def _model_decorator(
    make_model_func,
    *,
    library: ModelLibrary,
    flags: set[SupportFlag],
    keywords: list[str] | None = None,
    name: str | None = None,
):
    if name is None:
        name = make_model_func.__name__
    name = cast(str, name)

    if keywords is None:
        keywords = []

    check_name(name)
    if any(name == model.name for model in _models):
        raise ValueError(f"A model with name {name} is already registered")

    _models.append(
        ModelMaker(
            make_model=make_model_func,
            model_library=library,
            supported_flags=flags,
            keywords=keywords,
            name=name,
        )
    )
    return make_model_func


def _model(
    make_model_func=None,
    *,
    library: ModelLibrary,
    flags: set[SupportFlag],
    keywords: list[str] | None = None,
    name: str | None = None,
):
    if make_model_func is None:
        return partial(
            _model_decorator,
            library=library,
            flags=flags,
            keywords=keywords,
            name=name,
        )
    else:
        return _model_decorator(
            make_model_func,
            library=library,
            flags=flags,
            keywords=keywords,
            name=name,
        )


pymc_model = partial(_model, library="pymc", flags=set(["cuda", "jax", "numba"]))
pymc_model_jax = partial(_model, library="pymc", flags=set(["cuda", "jax"]))
stan_model = partial(_model, library="stan", flags=set())
