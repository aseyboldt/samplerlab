import argparse
import logging

import samplerlab
from samplerlab._stan_models import register_posteriordb


def setup_argparse():
    parser = argparse.ArgumentParser(
        prog="python -m samplerlab",
        description="Benchmark different samplers with various models.",
    )

    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        help="Specify keywords to select which models to use.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-s",
        "--samplers",
        nargs="+",
        help="Specify keywords to select which samplers to use.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="The seed for the sampler runs.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Specify the output directory where results will be stored.",
        required=True,
    )
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Store posteriors and effective sample sizes",
    )
    parser.add_argument(
        "--posteriordb",
        type=str,
        help="Location for a posteriordb repository.",
        required=False,
        default=None,
    )

    args = parser.parse_args()
    return args


def main():
    args = setup_argparse()

    if args.posteriordb is not None:
        register_posteriordb(args.posteriordb)

    models = samplerlab.select_models(args.models)
    samplers = samplerlab.select_samplers(args.samplers)
    samplerlab.filter_common_warnings()

    logger = logging.getLogger("pymc")
    logger.setLevel(logging.ERROR)

    success, failures = samplerlab.sample_models(
        args.seed,
        models,
        samplers,
        save_path=args.output,
        save_traces=args.save_traces,
    )

    print(f"Sampled {len(success)} models successfully, and failed {len(failures)}.")


if __name__ == "__main__":
    main()
