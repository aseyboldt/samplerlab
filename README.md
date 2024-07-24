# Run simple benchmarks and collect env data

To run the benchmarks, install pixi and run

```
pixi run benchmark
```

If you have an nvidia gpu, use

```
pixi run -e cuda12 benchmark
```

The collected data will be stored in the `traces` directory.

