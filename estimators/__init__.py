import torch
from .irls import IRLSSolver, IRLSFourier
from .gmm import GMMEstimator, RecursiveGMMEstimator
from .admm import ADMMSolver
from .weights import get_weight_function
from .distances import get_distance_function
from utils.space import Space
from typing import Dict


def build_estimator(
    method_cfg: dict, images: Dict[Space, torch.Tensor], device: str = "cpu"
):
    """
    Factory function that reads the YAML config block and returns
    the instantiated Estimator object on the specified device.
    """
    est_type = method_cfg["type"]
    params = method_cfg.get("params", {})

    if est_type == "m_estimator":
        params["solver_params"] = params.get("solver_params", {})
        params["solver_params"]["space"] = params["solver_params"].get("space", Space.REAL)
        if params["weight_function"] == "global" and (
            params.get("weight_params", None) is None
            or params["weight_params"].get("beta", "auto") == "auto"
        ):
            mult = params.get("weight_params", {}).get("auto_multiplier", 1)
            imgs = images[params["solver_params"]["space"]]
            beta = mult * 1.0e-5 / imgs.var(dim=(1, 2)).mean().item()
            est_name = method_cfg.get("name", None)
            if est_name is not None:
                print(f"Auto-calculated beta parameter for {est_name}: {beta = }")
            else:
                print(f"Auto-calculated beta parameter: {beta = }")
            weight_func = get_weight_function("global", {"beta": beta})
        else:
            weight_func = get_weight_function(
                params["weight_function"], params.get("weight_params", {})
            )
        return IRLSSolver(
            weight_function=weight_func,
            device=device,
            **params.get("solver_params", {}),
        )

    elif est_type == "fourier_m_estimator":
        config_real = params["real_estimator"]
        config_imag = params["imag_estimator"]
        config_real["params"]["solver_params"]["space"] = Space.FOURIER_REAL
        config_imag["params"]["solver_params"]["space"] = Space.FOURIER_IMAG
        irls_real = build_estimator(config_real, images, device)
        irls_imag = build_estimator(config_imag, images, device)
        return IRLSFourier(irls_real, irls_imag, device)

    elif est_type == "gmm":
        distance_func = get_distance_function(
            params["distance_function"], params.get("distance_params", {})
        )
        return GMMEstimator(
            distance_function=distance_func,
            random_state=params.get("random_state", None),
            device=device,
        )

    elif est_type == "recursive_gmm":
        distance_func = get_distance_function(
            params["distance_function"], params.get("distance_params", {})
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
