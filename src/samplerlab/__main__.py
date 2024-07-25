import argparse
import logging

import samplerlab


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

    args = parser.parse_args()
    return args


def main():
    args = setup_argparse()

    models = samplerlab.select_models(args.models)
    samplers = samplerlab.select_samplers(args.samplers)
    samplerlab.filter_common_warnings()

    logger = logging.getLogger("pymc")
    logger.setLevel(logging.ERROR)

    success, failures = samplerlab.sample_models(
        args.seed, models, samplers, save_path=args.output
    )

    print(f"Sampled {len(success)} models successfully, and failed {len(failures)}.")


if __name__ == "__main__":
    main()
