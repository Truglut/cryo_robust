import torch

from estimators.irls import (
    IRLSSolver,
    IRLSFourier,
    JointIRLSFourier,
    FlatteningIRLSFourier,
)
from estimators.gmm import GMMEstimator, RecursiveGMMEstimator
from estimators.admm import ADMMSolver
from estimators.weights import get_weight_function
from estimators.distances import get_distance_function

from method_comparison.domain.enums import Space

TAGARE_CONSTANT = 1.0e-5


def build_estimator(
    method_cfg: dict, images: dict[Space, torch.Tensor], device: str = "cpu", space: Space = Space.REAL
):
    """
    Factory function that reads the YAML config block and returns
    the instantiated Estimator object on the specified device.
    """
    est_type = method_cfg["type"]
    params = method_cfg.get("params", {})

    params["solver_params"] = params.get("solver_params", {})

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
        irls_real = build_estimator(config_real, images, device, space=Space.FOURIER_REAL)

        # Build imaginary part estimator
        config_imag = params["imag_estimator"]
        irls_imag = build_estimator(config_imag, images, device, space=Space.FOURIER_IMAG)

        # Build global Fourier estimator
        return IRLSFourier(irls_real, irls_imag, device)

    elif est_type == "joint_fourier":
        # Build IRLSSolver estimator
        solver = build_estimator(
            {
                "type": "m_estimator",
                "params": params,
            },
            images=images,
            device=device,
        )

        return JointIRLSFourier(solver, device)

    elif est_type == "flattening_fourier":
        solver = build_estimator(
            {
                "type": "m_estimator",
                "params": params,
            },
            images=images,
            device=device,
            space=Space.FOURIER_REAL
        )

        return FlatteningIRLSFourier(solver, device)

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
        irls_real = build_estimator(params["real_estimator"], images, device=device, space=Space.REAL)
        irls_fourier = build_estimator(
            params["fourier_estimator"], images, device=device, space=Space.FOURIER_REAL
        )
        return ADMMSolver(
            irls_real=irls_real,
            irls_fourier=irls_fourier,
            device=device,
            **params.get("solver_params", {}),
        )

    else:
        raise ValueError(f"Unknown estimator type: {est_type}")
