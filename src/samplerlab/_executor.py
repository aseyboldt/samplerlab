import json
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import arviz
import numpy as np
import watermark
from arviz import InferenceData
from rich.progress import Progress
from xarray import Dataset

from samplerlab._model_registry import ModelMaker, _models, check_name
from samplerlab._sampler_registry import Sampler, SampleResult, _samplers

__all__ = [
    "select_samplers",
    "select_models",
    "sample_models",
    "collect_system_data",
    "filter_common_warnings",
]


def select_samplers(keywords: list[str] | None = None) -> list[Sampler]:
    all_samplers = _samplers.copy()

    if keywords is not None:
        selection = []
        for sampler in all_samplers:
            if (
                any(word in sampler.keywords for word in keywords)
                or any(word in sampler.required_flags for word in keywords)
                or sampler.name in keywords
            ):
                selection.append(sampler)

        return selection
    else:
        return all_samplers


def select_models(keywords: list[str] | None = None) -> list[ModelMaker]:
    all_models = _models.copy()

    if keywords is not None:
        selection = []
        for model in all_models:
            if (
                any(word in model.keywords for word in keywords)
                or model.name in keywords
            ):
                selection.append(model)

        return selection
    else:
        return all_models


def _find_discrete(trace: InferenceData):
    is_discrete = ((trace.posterior == trace.posterior.round()).sum() > 100).to_array()
    if all(is_discrete):
        return []
    return list(
        is_discrete.where(is_discrete, np.nan).dropna("variable")["variable"].values
    )


@dataclass
class SamplingResult:
    model: ModelMaker
    sampler: Sampler
    result: SampleResult
    trace: InferenceData
    ess: Dataset
    rhat: Dataset
    stats: dict[str, Any]

    def name(self) -> str:
        return "-".join([self.sampler.name, *self.result.flags()])


@dataclass
class SamplingFailure:
    error: Exception
    model: ModelMaker
    sampler: Sampler

    def json(self):
        return {
            "error": str(self.error),
            "model": self.model.name,
            "sampler": self.sampler.name,
        }


def _postprocess_sampler_result(
    model_maker: ModelMaker,
    sampler: Sampler,
    result: SampleResult,
    save_directory: Path | None = None,
):
    trace = result.trace
    ess = arviz.ess(trace)
    rhat = arviz.rhat(trace)

    discrete = _find_discrete(trace)

    stats = dict(
        model_name=model_maker.name,
        sampler_name=sampler.name,
        wall_time=result.time_info.wall_time,
        process_time=result.time_info.process_time,
        float32=result.float32,
        device=result.device,
        meta=result.meta,
        min_effective_draws=float(
            ess.drop_vars(discrete).min().to_array().min().values
        ),
        max_effective_draws=float(
            ess.drop_vars(discrete).max().to_array().max().values
        ),
        max_rhat=float(rhat.max().to_array().max().values),
        max_rhat_no_discrete=float(
            rhat.drop_vars(discrete).max().to_array().max().values
        ),
    )

    if hasattr(trace, "sample_stats"):
        tr_stats = trace.sample_stats
        if hasattr(tr_stats, "n_steps"):
            stats["posterior_grad_evals"] = int(tr_stats.n_steps.sum())
        if hasattr(tr_stats, "acceptance_rate"):
            stats["acceptance_rate"] = float(tr_stats.acceptance_rate.mean())
            stats["acceptance_rate_geomean"] = float(
                np.exp(np.log(tr_stats.acceptance_rate).mean())
            )
        if hasattr(tr_stats, "mean_tree_accept"):
            stats["acceptance_rate"] = float(tr_stats.mean_tree_accept.mean())
            stats["acceptance_rate_geomean"] = float(
                np.exp(np.log(tr_stats.mean_tree_accept).mean())
            )
        if hasattr(tr_stats, "diverging"):
            stats["divergences"] = int(tr_stats.diverging.sum())

    if hasattr(trace, "warmup_sample_stats"):
        tr_stats = trace.warmup_sample_stats
        if hasattr(tr_stats, "n_steps"):
            stats["warmup_grad_evals"] = int(tr_stats.n_steps.sum())

    sampling_result = SamplingResult(
        trace=trace,
        ess=ess,
        rhat=rhat,
        sampler=sampler,
        model=model_maker,
        result=result,
        stats=stats,
    )

    if save_directory is not None:
        check_name(model_maker.name)
        check_name(sampler.name)
        final_dir = save_directory / model_maker.name / sampling_result.name()
        final_dir.mkdir(parents=True, exist_ok=False)
        trace.to_netcdf(final_dir / "trace.nc")
        ess.to_netcdf(final_dir / "ess.nc")
        rhat.to_netcdf(final_dir / "rhat.nc")

        with open(final_dir / "stats.json", "w") as file:
            json.dump(stats, file, indent=4)

    return sampling_result


def sample_models(
    seed: int,
    models: list[ModelMaker],
    samplers: list[Sampler],
    *,
    save_path: Path | str | None = None,
    progress_bar=False,
) -> tuple[list[SamplingResult], list[SamplingFailure]]:
    if save_path is not None:
        save_path = Path(save_path)
        save_path = save_path / datetime.now().strftime("%Y-%m-%dT%H%M%S")
        save_path.mkdir(parents=True, exist_ok=False)

    results = []
    failed = []
    items = list(product(models, samplers))
    progress = Progress(disable=not progress_bar)

    with progress:
        task = progress.add_task("Sampling models", total=len(items))
        for model, sampler in items:
            progress.print(f"Sampling {model.name} with {sampler.name}")
            try:
                result = sampler.sample_func(seed, model)
                sampler_result = _postprocess_sampler_result(
                    model, sampler, result, save_path
                )
            except Exception as err:
                failed.append(SamplingFailure(error=err, model=model, sampler=sampler))
            else:
                results.append(sampler_result)
            progress.advance(task)

    if save_path is not None:
        with (save_path / "summary.json").open("w") as file:
            json.dump(
                {
                    "system_info": collect_system_data(),
                    "sampling_runs": [result.stats for result in results],
                    "failures": [failure.json() for failure in failed],
                },
                file,
                indent=4,
            )

    return results, failed


def collect_system_data():
    import jax

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

    return {
        "numpy": numpy_info,
        "watermark": watermark_info,
        "nvidia-smi": output_smi,
        "jax": jax_info,
    }


def filter_common_warnings():
    warnings.filterwarnings(
        action="ignore", message="Explicitly requested dtype float64 requested"
    )
    warnings.filterwarnings(
        action="ignore", message="is incompatible with multithreaded code"
    )
    warnings.filterwarnings(
        action="ignore",
        message=r"os\.fork\(\) was called\. os\.fork\(\) is incompatible",
    )
