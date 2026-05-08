from typing import Dict

import torch

from .irls import IRLSSolver, IRLSFourier
from .gmm import GMMEstimator, RecursiveGMMEstimator
from .admm import ADMMSolver
from .weights import get_weight_function
from .distances import get_distance_function

from method_comparison.domain.enums import Space

TAGARE_CONSTANT = 1.0e-5


def build_estimator(
    method_cfg: dict, images: Dict[Space, torch.Tensor], device: str = "cpu"
):
    """
    Factory function that reads the YAML config block and returns
    the instantiated Estimator object on the specified device.
    """
    est_type = method_cfg["type"]
    params = method_cfg.get("params", {})

    params["solver_params"] = params.get("solver_params", {})
    space = params["solver_params"].get("space", Space.REAL)
    params["solver_params"]["space"] = space

    if est_type == "m_estimator":
        # Get weight function with given parameters
        weight_func = get_weight_function(
            params["weight_function"], params.get("weight_params", {}), images[space]
        )
        return IRLSSolver(
            weight_function=weight_func,
            device=device,
            **params.get("solver_params", {}),
        )

    elif est_type == "fourier_m_estimator":
        # Build real part estimator
        config_real = params["real_estimator"]
        config_real["params"]["solver_params"]["space"] = Space.FOURIER_REAL
        irls_real = build_estimator(config_real, images, device)

        # Build imaginary part estimator
        config_imag = params["imag_estimator"]
        config_imag["params"]["solver_params"]["space"] = Space.FOURIER_IMAG
        irls_imag = build_estimator(config_imag, images, device)

        # Build global Fourier estimator
        return IRLSFourier(irls_real, irls_imag, device)

    elif est_type == "gmm":
        distance_func = get_distance_function(
            params["distance_function"],
            params.get("distance_params", {}),
            images[space],
        )
        return GMMEstimator(
            distance_function=distance_func,
            random_state=params.get("random_state", None),
            device=device,
        )

    elif est_type == "recursive_gmm":
        distance_func = get_distance_function(
            params["distance_function"],
            params.get("distance_params", {}),
            images[space],
        )
        return RecursiveGMMEstimator(
            distance_function=distance_func,
            random_state=params.get("random_state", None),
            max_iter=params.get("max_iter", 10),
            tol=params.get("tol", 1e-3),
            device=device,
        )

    elif est_type == "admm":
        irls_real = build_estimator(params["real_estimator"], images, device=device)
        irls_fourier = build_estimator(
            params["fourier_estimator"], images, device=device
        )
        return ADMMSolver(
            irls_real, irls_fourier, device=device, **params.get("solver_params", {})
        )

    else:
        raise ValueError(f"Unknown estimator type: {est_type}")
