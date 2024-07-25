# Run simple benchmarks and collect env data

To run the benchmarks, install pixi and run

```
pixi run benchmark -o output_dir
```

If you have an nvidia gpu, use

```
pixi run -e cuda12 benchmark -o output_dir
```

The collected data will be stored in the `output_dir` directory.
The generated file `summary.json` contains summary statistics about the different sampling runs.

