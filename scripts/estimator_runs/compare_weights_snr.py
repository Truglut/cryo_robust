from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from cryo_robust.comparison.domain.enums import AggregationStrategy
from cryo_robust.comparison.evaluation.aggregation import aggregate_weights
from cryo_robust.domain import ImageSpace
from cryo_robust.estimators.admm import ADMMSolver
from cryo_robust.estimators.construction import build_estimator
from cryo_robust.estimators.data import ImageBatch
from cryo_robust.estimators.gmm import RecursiveGMMEstimator
from cryo_robust.estimators.results import WeightSet
from cryo_robust.utils.masks import create_fourier_mask
from scripts.estimator_runs.cli import build_simulation_parser
from scripts.estimator_runs.common import load_config
from scripts.estimator_runs.run_simulation import prepare_simulation_dataset


CORRECT_LABEL = 0


@dataclass
class MethodStatistics:
    """Statistics collected for one configured estimator."""

    quantity: str
    correct_means: list[float] = field(default_factory=list)
    correct_stds: list[float] = field(default_factory=list)
    incorrect_means: list[float] = field(default_factory=list)
    incorrect_stds: list[float] = field(default_factory=list)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    simulation_parser = build_simulation_parser()

    parser = argparse.ArgumentParser(
        description=(
            "Compare first-iteration estimator scores for correct and incorrect "
            "simulated images as a function of SNR. M-estimator-like methods use "
            "their first-iteration weights; recursive GMM methods use their raw "
            "first-iteration distances. All methods start from the sample mean."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML simulation configuration.",
    )
    parser.add_argument(
        "--snr",
        nargs="+",
        type=float,
        required=True,
        help="SNR values to evaluate.",
    )
    parser.add_argument(
        "--standardize",
        choices=["before", "after", "both", "none"],
        default=simulation_parser.get_default("standardize"),
        help="When to standardize generated images. Default: %(default)s.",
    )
    parser.add_argument(
        "--per-image-noise-std",
        action=argparse.BooleanOptionalAction,
        default=simulation_parser.get_default("per_image_noise_std"),
        help=(
            "Use an image-specific noise standard deviation, matching "
            "run_simulation. Default is inherited from run_simulation."
        ),
    )
    parser.add_argument(
        "--standardize-reference",
        action=argparse.BooleanOptionalAction,
        default=simulation_parser.get_default("standardize_reference"),
        help=(
            "Use the same reference-standardization option as run_simulation. "
            "Default is inherited from run_simulation."
        ),
    )
    parser.add_argument(
        "--fourier-weight-mask",
        choices=["low-pass", "band-pass", "high-pass", "none"],
        default=simulation_parser.get_default("fourier_weight_mask"),
        help=(
            "Fourier mask used when aggregating local Fourier weights, matching "
            "run_simulation. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=simulation_parser.get_default("device"),
        help="PyTorch device. Default: %(default)s.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override cfg['seed']. The same seed is reused for every SNR so that "
            "the simulated datasets differ only through the noise scale."
        ),
    )
    parser.add_argument(
        "--reference-image-path",
        type=Path,
        default=None,
        help="Override cfg['data']['reference_image_path'].",
    )
    parser.add_argument(
        "--misclassified-path",
        type=Path,
        default=None,
        help="Override cfg['data']['misclassified_path'].",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path at which to save the figure (for example, plot.png).",
    )

    return parser


def resolve_seed(cfg: dict, cli_seed: int | None) -> int:
    """Return the seed used for every SNR level."""
    if cli_seed is not None:
        return cli_seed

    config_seed = cfg.get("seed")
    if config_seed is not None:
        return int(config_seed)

    return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])


def validate_methods(method_configs: list[dict]) -> None:
    """Check that configured method names are present and unique."""
    names = [method.get("name") for method in method_configs]

    if any(name is None for name in names):
        raise ValueError("Every method in cfg['experiment']['methods'] needs a name.")

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Method names must be unique. Duplicates: {duplicates}")


def prepare_aggregation_masks(
    image_batch: ImageBatch,
    real_mask: np.ndarray | None,
    fourier_weight_mask: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Prepare the real- and Fourier-space masks used for weight aggregation."""
    real_mask_tensor = (
        None
        if real_mask is None
        else torch.as_tensor(real_mask, device=image_batch.device)
    )

    if fourier_weight_mask == "none":
        fourier_mask_tensor = None
    else:
        fourier_mask = create_fourier_mask(
            image_shape=tuple(image_batch.ensure_real().shape[-2:]),
            mask_type=fourier_weight_mask,
        )
        fourier_mask_tensor = torch.as_tensor(
            fourier_mask,
            device=image_batch.device,
        )

    return real_mask_tensor, fourier_mask_tensor


def canonical_aggregated_weights(
    weights: WeightSet,
    real_mask: torch.Tensor | None,
    fourier_mask: torch.Tensor | None,
) -> np.ndarray:
    """
    Reduce a WeightSet to one scalar score per image.

    The canonical-weight selection is delegated to WeightSet, as elsewhere in
    the repository: real-space weights have priority; otherwise Fourier real
    and imaginary weights are combined. Local weights are then reduced with
    the repository's mean aggregation function.
    """
    canonical = weights.canonical_weights()
    if canonical is None:
        raise ValueError("The estimator did not produce usable weights.")

    mask = real_mask if weights.real is not None else fourier_mask

    return aggregate_weights(
        canonical,
        strategy=AggregationStrategy.MEAN,
        mask=mask,
    )


@torch.inference_mode()
def first_admm_weights(
    estimator: ADMMSolver,
    image_batch: ImageBatch,
    reference_real: torch.Tensor,
) -> WeightSet:
    """
    Compute the weights from one ADMM outer step.

    Each IRLS subproblem is limited to one update so that the diagnostic really
    represents the initial weighting operation, before iterative refinement.
    """
    reference_fourier = torch.fft.rfft2(reference_real, norm=image_batch.norm)
    dual_vars = torch.zeros_like(reference_fourier)

    real_result, fourier_result, _ = estimator.step(
        image_batch,
        reference_real=reference_real,
        reference_fourier=reference_fourier,
        dual_vars=dual_vars,
        mu=estimator.initial_mu,
        real_irls_max_iter=1,
        fourier_irls_max_iter=1,
    )

    return WeightSet(
        real=real_result.weights.real,
        fourier_real=fourier_result.weights.fourier_real,
        fourier_imag=fourier_result.weights.fourier_imag,
    )


@torch.inference_mode()
def calculate_initial_scores(
    method_cfg: dict,
    image_batch: ImageBatch,
    real_mask: torch.Tensor | None,
    fourier_mask: torch.Tensor | None,
) -> tuple[np.ndarray, str]:
    """
    Return one initial score per image for one configured method.

    Recursive GMM estimators return the raw distances to the common initial
    reference. All other supported estimators return their first-iteration
    weights, reduced to one scalar per image with the repository aggregation
    logic.
    """
    estimator = build_estimator(method_cfg, image_batch=image_batch)

    # Keep the comparison controlled: every method starts from the same sample mean,
    # even if the normal experiment configuration specifies another initial reference.
    reference_real = image_batch.ensure_real().mean(dim=0)

    if isinstance(estimator, RecursiveGMMEstimator):
        distances = estimator.distance_function(
            image_batch.ensure_real(),
            reference_real,
        )
        scores = distances.reshape(-1).detach().cpu().numpy()
        quantity = "distance"

    elif isinstance(estimator, ADMMSolver):
        weights = first_admm_weights(
            estimator=estimator,
            image_batch=image_batch,
            reference_real=reference_real,
        )
        scores = canonical_aggregated_weights(
            weights,
            real_mask=real_mask,
            fourier_mask=fourier_mask,
        )
        quantity = "weight"

    else:
        # IRLSSolver, IRLSFourier, JointIRLSFourier and FlatteningIRLSFourier
        # all accept max_iter_override. This evaluates exactly one weighting/update
        # step while letting each estimator handle its native data representation.
        result = estimator.fit(
            image_batch,
            reference=reference_real,
            max_iter_override=1,
        )
        scores = canonical_aggregated_weights(
            result.weights,
            real_mask=real_mask,
            fourier_mask=fourier_mask,
        )
        quantity = "weight"

    if scores.shape != (image_batch.n_images,):
        raise ValueError(
            f"Method {method_cfg['name']!r} produced scores with shape "
            f"{scores.shape}; expected ({image_batch.n_images},)."
        )

    return scores, quantity


def summarize_scores(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float, float, int, int]:
    """Return mean/std for correct images and for all outlier labels together."""
    correct = labels == CORRECT_LABEL
    incorrect = ~correct

    if not np.any(correct):
        raise ValueError("The generated dataset contains no correct images (label 0).")
    if not np.any(incorrect):
        raise ValueError(
            "The generated dataset contains no incorrect images (labels != 0)."
        )

    correct_scores = scores[correct]
    incorrect_scores = scores[incorrect]

    return (
        float(correct_scores.mean()),
        float(correct_scores.std()),
        float(incorrect_scores.mean()),
        float(incorrect_scores.std()),
        int(correct.sum()),
        int(incorrect.sum()),
    )


def plot_statistics(
    snr_values: np.ndarray,
    statistics: dict[str, MethodStatistics],
    output: Path | None,
) -> None:
    """Plot score means and standard deviations versus SNR in two subplots."""
    order = np.argsort(snr_values)
    x = snr_values[order]

    fig, (mean_ax, std_ax) = plt.subplots(
        2,
        1,
        figsize=(10.0, 8.0),
        sharex=True,
    )

    has_weights = any(stats.quantity == "weight" for stats in statistics.values())
    has_distances = any(stats.quantity == "distance" for stats in statistics.values())

    mean_distance_ax = mean_ax.twinx() if has_distances else None
    std_distance_ax = std_ax.twinx() if has_distances else None

    method_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    legend_handles: list[Line2D] = []

    for idx, (method_name, stats) in enumerate(statistics.items()):
        color = method_colors[idx % len(method_colors)]

        if stats.quantity == "distance":
            mean_target = mean_distance_ax
            std_target = std_distance_ax
            legend_name = f"{method_name} [distance]"
        else:
            mean_target = mean_ax
            std_target = std_ax
            legend_name = f"{method_name} [weight]"

        assert mean_target is not None
        assert std_target is not None

        correct_means = np.asarray(stats.correct_means)[order]
        incorrect_means = np.asarray(stats.incorrect_means)[order]
        correct_stds = np.asarray(stats.correct_stds)[order]
        incorrect_stds = np.asarray(stats.incorrect_stds)[order]

        mean_target.plot(
            x,
            correct_means,
            marker="o",
            linestyle="-",
            color=color,
        )
        mean_target.plot(
            x,
            incorrect_means,
            marker="o",
            linestyle="--",
            color=color,
        )

        std_target.plot(
            x,
            correct_stds,
            marker="o",
            linestyle="-",
            color=color,
        )
        std_target.plot(
            x,
            incorrect_stds,
            marker="o",
            linestyle="--",
            color=color,
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker="o",
                linestyle="-",
                label=legend_name,
            )
        )

    mean_ax.set_title("Mean first-iteration score")
    std_ax.set_title("Standard deviation of first-iteration score")

    if has_weights:
        mean_ax.set_ylabel("Aggregated weight")
        std_ax.set_ylabel("Aggregated weight")

    if mean_distance_ax is not None:
        mean_distance_ax.set_ylabel("GMM distance")
    if std_distance_ax is not None:
        std_distance_ax.set_ylabel("GMM distance")

    std_ax.set_xlabel("Signal-to-noise ratio (SNR)")

    mean_ax.grid(alpha=0.3)
    std_ax.grid(alpha=0.3)

    group_handles = [
        Line2D([0], [0], color="black", linestyle="-", label="Correct images"),
        Line2D([0], [0], color="black", linestyle="--", label="Incorrect images"),
    ]

    ncols = min(3, max(1, len(legend_handles)))
    fig.legend(
        handles=legend_handles + group_handles,
        loc="lower center",
        ncol=ncols,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 1.0))

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"\nSaved figure to: {output}")

    plt.show()


def main() -> None:
    """Run the first-iteration comparison for every configured estimator."""
    args = build_parser().parse_args()

    if any(snr <= 0 for snr in args.snr):
        raise ValueError("All SNR values must be strictly positive.")

    cfg = load_config(
        args.config,
        reference_image_path=args.reference_image_path,
        misclassified_path=args.misclassified_path,
    )
    method_configs = cfg.get("experiment", {}).get("methods", [])
    if not method_configs:
        raise ValueError("cfg['experiment']['methods'] contains no methods.")

    validate_methods(method_configs)

    seed = resolve_seed(cfg, args.seed)
    statistics: dict[str, MethodStatistics] = {}

    print(f"Using seed: {seed}")
    print(
        "All methods use the mean of all estimator images as their common "
        "initial reference."
    )
    print(
        "Incorrect images are defined as every simulated image with label != 0 "
        "(very rotated, misclassified, or pure-noise images)."
    )
    print(
        "Weight methods: first weighting/update only. "
        "GMM methods: raw distances before GMM standardization/fitting.\n"
    )

    header = (
        f"{'SNR':>10}  {'method':<28}  {'quantity':<8}  "
        f"{'N correct':>9}  {'correct mean':>12}  {'correct std':>11}  "
        f"{'N incorrect':>11}  {'incorrect mean':>14}  {'incorrect std':>13}"
    )
    print(header)
    print("-" * len(header))

    for snr in args.snr:
        # Reset the RNG for each SNR so the underlying generated examples and
        # normalized random draws stay paired across noise levels.
        rng = np.random.default_rng(seed)

        dataset = prepare_simulation_dataset(
            cfg,
            snr=snr,
            rng=rng,
            standardize=args.standardize,
            per_image_noise_std=args.per_image_noise_std,
            standardize_reference=args.standardize_reference,
            device=args.device,
        )
        image_batch = ImageBatch.from_real(dataset.estimator_images)

        real_mask, fourier_mask = prepare_aggregation_masks(
            image_batch=image_batch,
            real_mask=dataset.mask,
            fourier_weight_mask=args.fourier_weight_mask,
        )

        for method_cfg in method_configs:
            method_name = method_cfg["name"]

            scores, quantity = calculate_initial_scores(
                method_cfg=method_cfg,
                image_batch=image_batch,
                real_mask=real_mask,
                fourier_mask=fourier_mask,
            )
            (
                correct_mean,
                correct_std,
                incorrect_mean,
                incorrect_std,
                n_correct,
                n_incorrect,
            ) = summarize_scores(scores, dataset.labels)

            if method_name not in statistics:
                statistics[method_name] = MethodStatistics(quantity=quantity)
            elif statistics[method_name].quantity != quantity:
                raise RuntimeError(
                    f"Method {method_name!r} changed score type across SNR levels."
                )

            method_stats = statistics[method_name]
            method_stats.correct_means.append(correct_mean)
            method_stats.correct_stds.append(correct_std)
            method_stats.incorrect_means.append(incorrect_mean)
            method_stats.incorrect_stds.append(incorrect_std)

            print(
                f"{snr:10.5g}  {method_name:<28.28}  {quantity:<8}  "
                f"{n_correct:9d}  {correct_mean:12.6f}  {correct_std:11.6f}  "
                f"{n_incorrect:11d}  {incorrect_mean:14.6f}  "
                f"{incorrect_std:13.6f}"
            )

    plot_statistics(
        snr_values=np.asarray(args.snr, dtype=float),
        statistics=statistics,
        output=args.output,
    )


if __name__ == "__main__":
    main()
