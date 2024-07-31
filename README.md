# Benchmark MCMC samplers on different models

## Running existing benchmarks

To run the benchmarks, install pixi and run

```
pixi run benchmark -o output_dir
```

If you have an nvidia gpu, use

```
pixi run -e cuda12 benchmark -o output_dir
```

Show more command line arguments:

```
pixi run benchmark -h
```

If for instance you want to run all registered stan samplers using models from posteriordb,
you have first to clone the posteriordb repository and then select posteriordb models:

```
git clone https://github.com/stan-dev/posteriordb.git
pixi run benchmark --posteriordb posteriordb/posterior_database -m posteriordb --save-traces -o traces-stan
```

Replace `-m posteriordb` by `-m posteriordb-fast` to exclude some very slow models.

This will save the posteriors of the different samplers to the directory `traces-stan`.

To read the summary statistics you can use code like the following:

```python
import pathlib
import pandas as pd

base = pathlib.Path("traces-stan/<replace-timestamp>/")
data = []

for file in base.glob("**/stats.json"):
    with file.open() as file:
        data.append(json.load(file))

df = pd.json_normalize(data)

df["seconds_per_ess"] = df["wall_time"] / df["min_effective_draws"]
```


The generated file `summary.json` contains summary statistics about the different sampling runs, failures and some info about the benchmark machine.


## Adding models or samplers

You can register new samplers or models using the `pymc_sampler`, `stan_sampler`, `pymc_model` and `stan_model` decorators.

pymc model functions need to return a pymc model, stan model function need to return a tuple of model code and dataset.

You can find examples in the `_samplers.py`, `_models.py` or `_stan_models.py` files.

You can then run the benchmark pipeline manually using the `select_models`, `select_sampler` and `sample_models` functions.
