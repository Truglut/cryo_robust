import argparse
from pathlib import Path

from cryo_robust.comparison.visualization.plotting import BASE_PLOT_OPTIONS

ALL_PLOTS = ["weights", "gmm", "frc"]


def build_base_parser() -> tuple[
    argparse.ArgumentParser,
    argparse._ArgumentGroup,
    argparse._ArgumentGroup,
    argparse._ArgumentGroup,
]:
    """
    Build the base command-line argument parser, with arguments that are common
    to simulated data and experimental data runs
    """
    parser = argparse.ArgumentParser(description="Run robust estimators on real data")
    visualization_group = parser.add_argument_group("Visualization")
    subset_group = parser.add_argument_group("Subset selection")
    saving_group = parser.add_argument_group("Saving")
    evaluation_group = parser.add_argument_group("Evaluation")

    # Global arguments
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Compute device for PyTorch"
    )

    # Visualization
    visualization_group.add_argument(
        "--plot",
        nargs="+",
        type=str,
        choices=ALL_PLOTS + ["all"],
        help="Plots to generate",
    )
    visualization_group.add_argument(
        "--print",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the report to terminal"
    )
    visualization_group.add_argument(
        "--max-subplots",
        type=int,
        default=BASE_PLOT_OPTIONS["max_subplots"],
        help="Maximum number of subplots to include in the same figure",
    )
    visualization_group.add_argument(
        "--dpi",
        type=int,
        default=BASE_PLOT_OPTIONS["dpi"],
        help="DPI for saved figures",
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

    # Evaluation
    evaluation_group.add_argument(
        "--reapply-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reapply the mask to estimations before evaluation"
    )
    evaluation_group.add_argument(
        "--independent-half-sets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Independently re-fit estimators on half sets for half-set FRC calculations"
    )
    evaluation_group.add_argument(
        "--fourier-weight-mask",
        choices=["low-pass", "band-pass", "high-pass", "none"],
        help="Type of mask to be used when evaluating weights in Fourier Space",
        default="none"
    )
    return parser, visualization_group, subset_group, saving_group, evaluation_group


def build_simulation_parser() -> argparse.ArgumentParser:
    """
    Builds the simulation argument parser by adding the simulation-specific
    arguments to the base parser.
    """
    parser, _, _, saving_group, _ = build_base_parser()

    # Add reports to saving group
    saving_group.add_argument(
        "--report", type=Path, help="Generate a LaTeX report at the provided path"
    )

    simulation_group = parser.add_argument_group("Simulation")

    # Simulation
    simulation_group.add_argument(
        "--snr",
        nargs="+",
        type=float,
        help="Target signal to noise ratio in image generation",
    )

    simulation_group.add_argument(
        "--standardize",
        choices=["before", "after", "both", "none"],
        default="after",
        help="When to standardize generated images. Default is %(default)s"
    )
    simulation_group.add_argument(
        "--per-image-noise-std",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Add a different noise std to each image to achieve a uniform SNR. Default is False"
    )
    simulation_group.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of simulations to run with the specified configuration."
    )
    return parser


def build_experimental_parser() -> argparse.ArgumentParser:
    """
    Builds the experimental image estimator runs argument parser by adding 
    its specific arguments to the base parser.
    """
    parser, _, _, _, _ = build_base_parser()
    parser.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable per-image standardization",
    )
    return parser


def parse_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parses the command-line arguments, performing some basic input validation
    """
    args = parser.parse_args()

    # Input validation
    if args.save_quantiles and not args.quantiles:
        parser.error("--save-quantiles requires --quantiles")

    if args.save_thresholds and not args.thresholds:
        parser.error("--save-thresholds requires --thresholds")

    # Standardize args.plot to always be a list
    if args.plot is None:
        args.plot = []
    if "all" in args.plot:
        args.plot = ALL_PLOTS

    # Build plot options
    args.plot_options = {
        "max_subplots": args.max_subplots,
        "density": args.density,
        "dpi": args.dpi,
    }

    return args


def quantile(value) -> float:
    """
    Input validation for quantiles
    """
    q = float(value)
    if not (0 <= q <= 1):
        raise argparse.ArgumentTypeError("Quantiles must be in [0, 1]")

    return q
