from .irls import MEstimator
from .gmm import GMMEstimator, RecursiveGMMEstimator
from .weights import get_weight_function
from .distances import get_distance_function


def build_estimator(method_cfg: dict, device: str = "cpu"):
    """
    Factory function that reads the YAML config block and returns
    the instantiated Estimator object on the specified device.
    """
    est_type = method_cfg["type"]
    params = method_cfg.get("params", {})

    if est_type == "m_estimator":
        weight_func = get_weight_function(
            params["weight_function"], params.get("weight_params", {})
        )
        return MEstimator(
            weight_function=weight_func,
            max_iter=params.get("max_iter", 100),
            device=device,  # Pass the device here
        )

    elif est_type == "gmm":
        distance_func = get_distance_function(
            params["distance_function"], params.get("distance_params", {})
        )
        return GMMEstimator(
            distance_function=distance_func,
            random_state=params.get("random_state", None),
            device=device
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
            device=device
        )

    else:
        raise ValueError(f"Unknown estimator type: {est_type}")
