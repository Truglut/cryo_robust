import argparse
from pathlib import Path

from method_comparison.dataset_builder import STANDARDIZE_TYPES
from method_comparison.visualization.plotting import BASE_PLOT_OPTIONS

BASE_PLOTS = ["weights", "gmm"]
SIMULATION_PLOTS = ["fsc"]
EXPERIMENTAL_PLOTS = []


def build_base_parser() -> tuple[
    argparse.ArgumentParser,
    argparse._ArgumentGroup,
    argparse._ArgumentGroup,
    argparse._ArgumentGroup,
]:
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
        "--max-subplots",
        type=int,
        default=BASE_PLOT_OPTIONS["max_subplots"],
        help="Maximum number of subplots to include in the same figure",
    )
    visualization_group.add_argument(
        "--dpi", type=int, default=BASE_PLOT_OPTIONS["dpi"], help="DPI for saved figures"
    )
    visualization_group.add_argument(
        "--density",
        action="store_true",
        help="Normalize weight distribution histograms to probability densities",
    )
    visualization_group.add_argument(
        "--show-images",
        action="store_true",
        help="Show generated images",
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
    parser, visualization_group, _, saving_group = build_base_parser()

    # Add relevant plots to visualization
    visualization_group.add_argument(
        "--plot",
        nargs="+",
        type=str,
        choices=BASE_PLOTS + SIMULATION_PLOTS,
        help="Plots to generate",
    )

    # Add reports to saving group
    saving_group.add_argument(
        "--report", type=Path, help="Generate a LaTeX report at the provided path"
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
        "--standardize",
        type=str,
        choices=STANDARDIZE_TYPES,
        help="Standardize images *after* adding noise. By default no standardization is done",
    )

    # Evaluation
    evaluation_group.add_argument(
        "--reapply-mask",
        action="store_true",
        help="Reapply the mask to estimations before evaluation",
    )
    return parser


def build_experimental_parser() -> argparse.ArgumentParser:
    parser, visualization_group, _, _ = build_base_parser()

    # Add relevant plots to visualization group
    visualization_group.add_argument(
        "--plot",
        nargs="+",
        type=str,
        choices=BASE_PLOTS + EXPERIMENTAL_PLOTS,
        help="Plots to generate",
    )
    return parser


def parse_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()

    # Input validation
    if args.save_quantiles and not args.quantiles:
        parser.error("--save-quantiles requires --quantiles")

    if args.save_thresholds and not args.thresholds:
        parser.error("--save-thresholds requires --thresholds")

    # Standardize args.plot to always be a list
    if args.plot is None:
        args.plot = []

    # Build plot options
    args.plot_options = {
        "max_subplots": args.max_subplots,
        "density": args.density,
        "dpi": args.dpi,
    }

    return args


def quantile(value) -> float:
    """Input validation for quantiles"""
    q = float(value)
    if not (0 <= q <= 1):
        raise argparse.ArgumentTypeError("Quantiles must be in [0, 1]")

    return q
