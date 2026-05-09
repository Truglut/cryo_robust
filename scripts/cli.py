import argparse
from pathlib import Path


BASE_PLOTS = ["weights", "gmm"]
SIMULATION_PLOTS = ["fsc"]
REAL_PLOTS = []


def build_base_parser() -> argparse.ArgumentParser:
    """Parses the config from the command line"""
    parser = argparse.ArgumentParser(description="Run robust estimators on real data")
    visualization_group = parser.add_argument_group("Visualization")
    subset_group = parser.add_argument_group("Subset selection")
    saving_group = parser.add_argument_group("Saving")

    # Global arguments
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Compute device for PyTorch"
    )

    # Visualization
    visualization_group.add_argument(
        "--show-images",
        action="store_true",
        help="Show generated images",
    )
    visualization_group.add_argument(
        "--gmm-evaluation",
        action="store_true",
        help="Show the initial and final fits of GMM models",
    )
    visualization_group.add_argument(
        "--plot-weights",
        action="store_true",
        help="Show plots of weight distributions for each estimation method",
    )

    # Subset selection
    subset_group.add_argument(
        "--quantiles",
        type=quantile,
        nargs="+",
        help="Quantiles for which to show (and optionally save with --save-quantiles) best and worst images",
    )
    subset_group.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        help="Weight thresholds for which to show (and optionally save with --save-thresholds) good and bad images",
    )

    # Saving
    saving_group.add_argument(
        "--save-quantiles",
        action="store_true",
        help="Save images with highest and lowest weights for each quantile in --quantiles",
    )
    saving_group.add_argument(
        "--save-thresholds",
        action="store_true",
        help="Save images with weights higher and lower than each given threshold",
    )
    saving_group.add_argument(
        "--save-unaligned",
        action="store_true",
        help="Any images saved will be the original, unaligned images",
    )
    saving_group.add_argument(
        "--save-weights",
        action="store_true",
        help="Save final weights for every estimation method as a .npy file",
    )
    return parser, visualization_group, subset_group, saving_group


def build_simulation_parser() -> argparse.ArgumentParser:
    parser, visualization_group, _, _ = build_base_parser()

    # Visualization: add specific plots
    visualization_group.add_argument(
        "--plot-fsc",
        action="store_true",
        help="Plot FSC resolution for all methods (overlayed on one figure)",
    )

    simulation_group = parser.add_argument_group("Simulation")
    evaluation_group = parser.add_argument_group("Evaluation")

    # Simulation
    simulation_group.add_argument(
        "--snr",
        type=float,
        help="Target signal to noise ratio in image generation. Overrides snr in config file",
    )
    simulation_group.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize images to [0,1] before adding noise/rotating",
    )

    # Evaluation
    evaluation_group.add_argument(
        "--reapply-mask",
        action="store_true",
        help="Reapply the mask to estimations before evaluation",
    )
    return parser


def build_experimental_parser() -> argparse.ArgumentParser:
    parser, _, _, _ = build_base_parser()
    return parser


def parse_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()

    # Input validation
    if args.save_quantiles and not args.quantiles:
        parser.error("--save-quantiles requires --quantiles")

    if args.save_thresholds and not args.thresholds:
        parser.error("--save-thresholds requires --thresholds")

    return args


def quantile(value) -> float:
    """Input validation for quantiles"""
    q = float(value)
    if not (0 <= q <= 1):
        raise argparse.ArgumentTypeError("Quantiles must be in [0, 1]")

    return q
