from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import napari
import numpy as np

from cryo_robust.comparison.dataset_builder import LABEL_TYPES
from scripts.estimator_runs.cli import build_simulation_parser
from scripts.estimator_runs.common import load_config
from scripts.estimator_runs.run_simulation import prepare_simulation_dataset

# ---------------------------------------------------------------------------
# Viewer defaults
# ---------------------------------------------------------------------------
# Number of examples shown for each image category. It can be overridden from
# the CLI with --n-examples.
DEFAULT_N_EXAMPLES = 10


STANDARDIZATION_CHOICES = ("before", "after", "both", "none")

REFERENCE_LABEL = 0
VERY_ROTATED_LABEL = 1
MISCLASSIFIED_LABEL = 2
NOISE_LABEL = 3


@dataclass(frozen=True)
class DatasetVariant:
    """Description of one simulated dataset variant."""

    snr: float
    standardize: str
    per_image_noise_std: bool

    @property
    def noise_name(self) -> str:
        return "per-image" if self.per_image_noise_std else "global"

    @property
    def name(self) -> str:
        return (
            f"SNR={self.snr:g} | noise={self.noise_name} "
            f"| standardize={self.standardize}"
        )


@dataclass
class GeneratedDataset:
    """Arrays associated with one generated dataset variant."""

    images: np.ndarray
    ground_truth: np.ndarray
    labels: np.ndarray
    unmasked_images: np.ndarray | None = None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for dataset inspection."""
    simulation_parser = build_simulation_parser()

    parser = argparse.ArgumentParser(
        description=(
            "Generate and visually inspect the same simulated datasets used by "
            "scripts.estimator_runs.run_simulation, without running estimators."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the same YAML configuration used by run_simulation.",
    )
    parser.add_argument(
        "--snr",
        nargs="+",
        type=float,
        default=simulation_parser.get_default("snr"),
        help=(
            "Target SNR value(s). The default is inherited from run_simulation: "
            "%(default)s."
        ),
    )
    parser.add_argument(
        "--standardize",
        choices=STANDARDIZATION_CHOICES,
        default=simulation_parser.get_default("standardize"),
        help=(
            "Standardization mode for the base dataset. The default is inherited "
            "from run_simulation: %(default)s."
        ),
    )
    parser.add_argument(
        "--per-image-noise-std",
        action=argparse.BooleanOptionalAction,
        default=simulation_parser.get_default("per_image_noise_std"),
        help=(
            "Use an image-specific noise standard deviation for the base dataset. "
            "The default is inherited from run_simulation."
        ),
    )
    parser.add_argument(
        "--standardize-reference",
        action=argparse.BooleanOptionalAction,
        default=simulation_parser.get_default("standardize_reference"),
        help=(
            "Use the same reference-standardization option as run_simulation. "
            "The default is inherited from run_simulation."
        ),
    )

    comparison_group = parser.add_argument_group("Additional variants")
    comparison_group.add_argument(
        "--compare-noise",
        action="store_true",
        help=(
            "Also generate the same dataset with the opposite noise mode. "
            "With the default base dataset this adds the per-image-noise variant."
        ),
    )
    comparison_group.add_argument(
        "--add-standardize",
        nargs="+",
        choices=STANDARDIZATION_CHOICES,
        default=[],
        metavar="MODE",
        help=(
            "Additional standardization modes to inspect. If --compare-noise is "
            "also used, both noise modes are generated for every requested mode."
        ),
    )

    display_group = parser.add_argument_group("Display")
    display_group.add_argument(
        "--n-examples",
        type=int,
        default=DEFAULT_N_EXAMPLES,
        help=(
            "Number of examples to show for each image category "
            "(reference copies, very rotated copies, misclassified outliers and "
            f"noise images). Default: {DEFAULT_N_EXAMPLES}."
        ),
    )
    display_group.add_argument(
        "--show-unmasked",
        action="store_true",
        help=(
            "Also add unmasked versions of the displayed image subsets. "
            "Without this flag only the masked images actually passed to the "
            "estimators are shown."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override cfg['seed']. All displayed variants use the same seed so "
            "that their random rotations, sampled outliers and Gaussian draws are "
            "directly comparable."
        ),
    )

    return parser


def build_variants(args: argparse.Namespace) -> list[DatasetVariant]:
    """Build requested combinations of SNR, standardization and noise mode."""
    standardizations = [args.standardize]
    standardizations.extend(
        mode for mode in args.add_standardize if mode not in standardizations
    )

    noise_modes = [args.per_image_noise_std]
    if args.compare_noise:
        noise_modes.append(not args.per_image_noise_std)

    return [
        DatasetVariant(
            snr=snr,
            standardize=standardize,
            per_image_noise_std=per_image_noise_std,
        )
        for snr in args.snr
        for standardize in standardizations
        for per_image_noise_std in noise_modes
    ]


def resolve_seed(cfg: dict, cli_seed: int | None) -> int:
    """Return one seed shared by every displayed variant."""
    if cli_seed is not None:
        return cli_seed

    config_seed = cfg.get("seed")
    if config_seed is not None:
        return int(config_seed)

    # run_simulation is non-deterministic when cfg["seed"] is None. For this
    # comparison tool, choose one seed and reuse it across variants so that
    # differences are caused by the requested options rather than by resampling.
    return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])


def generate_variant(
    cfg: dict,
    variant: DatasetVariant,
    seed: int,
    *,
    standardize_reference: bool,
    keep_unmasked: bool,
) -> GeneratedDataset:
    """Generate one dataset through the run_simulation preprocessing pipeline."""
    # Restarting the RNG for each variant makes comparisons much easier:
    # rotations, selected outliers and Gaussian draws stay aligned.
    rng = np.random.default_rng(seed)

    prepared = prepare_simulation_dataset(
        cfg,
        snr=variant.snr,
        rng=rng,
        standardize=variant.standardize,
        per_image_noise_std=variant.per_image_noise_std,
        standardize_reference=standardize_reference,
        device="cpu",
        keep_unmasked=keep_unmasked,
    )

    return GeneratedDataset(
        images=prepared.estimator_images.detach().cpu().numpy(),
        ground_truth=prepared.ground_truth,
        labels=prepared.labels,
        unmasked_images=prepared.unmasked_images,
    )


def print_dataset_summary(variant: DatasetVariant, dataset: GeneratedDataset) -> None:
    """Print a compact numerical summary for one generated dataset."""
    images = dataset.images
    finite = np.isfinite(images)
    n_nonfinite = images.size - int(finite.sum())

    print(f"\n{variant.name}")
    print(f"  shape: {images.shape}")
    print(f"  non-finite values: {n_nonfinite}")

    label_counts = {
        LABEL_TYPES.get(int(label), str(int(label))): int(
            (dataset.labels == label).sum()
        )
        for label in np.unique(dataset.labels)
    }
    print(f"  labels: {label_counts}")

    if n_nonfinite:
        print(
            "  WARNING: this variant contains NaN/inf values. "
            "They are intentionally preserved so that problems in the simulation "
            "pipeline remain visible."
        )


def safe_mean(images: np.ndarray) -> np.ndarray:
    """Compute a visualization mean while tolerating non-finite values."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(images, axis=0)


def add_subset(
    viewer: napari.Viewer,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    label: int,
    n_examples: int,
    name: str,
) -> None:
    """Add up to ``n_examples`` images from one label as a napari layer."""
    if n_examples <= 0:
        return

    subset = images[labels == label]
    n_show = min(n_examples, subset.shape[0])
    if n_show == 0:
        return

    viewer.add_image(
        subset[:n_show],
        name=f"{name} ({n_show})",
        visible=False,
    )


def add_variant_to_viewer(
    viewer: napari.Viewer,
    variant: DatasetVariant,
    dataset: GeneratedDataset,
    *,
    n_examples: int,
    show_unmasked: bool,
    first_variant: bool,
) -> None:
    """Add a compact set of dataset summaries and examples to napari."""
    prefix = variant.name

    viewer.add_image(
        dataset.ground_truth,
        name=f"{prefix} | ground truth",
        visible=first_variant,
    )

    viewer.add_image(
        safe_mean(dataset.images),
        name=f"{prefix} | mean of all images",
        visible=False,
    )

    inlier_mask = dataset.labels == REFERENCE_LABEL
    if np.any(inlier_mask):
        viewer.add_image(
            safe_mean(dataset.images[inlier_mask]),
            name=f"{prefix} | mean of inliers",
            visible=False,
        )

    subset_specs = (
        (REFERENCE_LABEL, LABEL_TYPES[REFERENCE_LABEL]),
        (VERY_ROTATED_LABEL, LABEL_TYPES[VERY_ROTATED_LABEL]),
        (MISCLASSIFIED_LABEL, LABEL_TYPES[MISCLASSIFIED_LABEL]),
        (NOISE_LABEL, LABEL_TYPES[NOISE_LABEL]),
    )

    for label, description in subset_specs:
        add_subset(
            viewer,
            dataset.images,
            dataset.labels,
            label=label,
            n_examples=n_examples,
            name=f"{prefix} | {description}",
        )

    if not show_unmasked or dataset.unmasked_images is None:
        return

    # Only when explicitly requested, add unmasked versions of the representative
    # subsets. Means remain based on the actual masked estimator input.
    for label, description in subset_specs:
        add_subset(
            viewer,
            dataset.unmasked_images,
            dataset.labels,
            label=label,
            n_examples=n_examples,
            name=f"{prefix} | {description} | unmasked",
        )


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate display-related CLI arguments."""
    if args.n_examples < 0:
        parser.error("--n-examples must be >= 0")


def main() -> None:
    """Generate requested simulated datasets and launch napari."""
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    cfg = load_config(args.config)
    variants = build_variants(args)
    seed = resolve_seed(cfg, args.seed)

    print(f"Using comparison seed: {seed}")
    print(f"Generating {len(variants)} dataset variant(s).")

    viewer = napari.Viewer(title="cryo_robust simulated dataset inspector")

    for i, variant in enumerate(variants):
        dataset = generate_variant(
            cfg,
            variant,
            seed,
            standardize_reference=args.standardize_reference,
            keep_unmasked=args.show_unmasked,
        )
        print_dataset_summary(variant, dataset)

        add_variant_to_viewer(
            viewer,
            variant,
            dataset,
            n_examples=args.n_examples,
            show_unmasked=args.show_unmasked,
            first_variant=(i == 0),
        )

    napari.run()


if __name__ == "__main__":
    main()
