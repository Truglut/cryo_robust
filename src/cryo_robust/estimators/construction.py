from typing import Literal, Callable, get_args
from copy import deepcopy

from cryo_robust.domain import ImageSpace
from cryo_robust.estimators.data import ImageBatch

from cryo_robust.estimators.base import Estimator
from cryo_robust.estimators.admm import ADMMSolver
from cryo_robust.estimators.irls import IRLSSolver
from cryo_robust.estimators.fourier_irls import (
    FlatteningIRLSFourier,
    IRLSFourier,
    JointIRLSFourier,
)
from cryo_robust.estimators.gmm import RecursiveGMMEstimator, FourierGMM
from cryo_robust.estimators.fourier_masked import MaskedIRLSFourier, ThreeMasksFourier

from cryo_robust.estimators.distances import get_distance_function
from cryo_robust.estimators.weights import (
    get_weight_function,
    norm_smooth_redescending_weights,
)

from cryo_robust.utils.masks import create_lowpass_rfft_mask

LOWPASS_NORMALIZED_CUTOFF = 0.3
DEFAULT_FOURIER_GMM_DELTA = 1.5

EstimatorTypeName = Literal[
    "m_estimator",
    "fourier_m_estimator",
    "joint_fourier",
    "flattening_fourier",
    "recursive_gmm",
    "admm",
    "masked_fourier",
    "three_masks_fourier",
    "fourier_gmm",
]


def _build_m_estimator(
    params: dict, image_batch: ImageBatch, space: ImageSpace
) -> IRLSSolver:
    weight_func = get_weight_function(
        name=params["weight_function"],
        params=params.get("weight_params", {}),
        imgs=image_batch.select_space(space),
    )
    params["solver_params"]["space"] = space
    return IRLSSolver(weight_function=weight_func, **params.get("solver_params", {}))


def _build_fourier_m_estimator(
    params: dict, image_batch: ImageBatch, space: ImageSpace
) -> IRLSFourier:
    # Build real part estimator
    config_real = params["real_estimator"]
    irls_real = build_estimator(config_real, image_batch, ImageSpace.FOURIER_REAL)

    # Build imaginary part estimator
    config_imag = params["imag_estimator"]
    irls_imag = build_estimator(config_imag, image_batch, ImageSpace.FOURIER_IMAG)

    # Build global Fourier estimator
    return IRLSFourier(irls_real, irls_imag)


def _build_joint_fourier(
    params: dict, image_batch: ImageBatch, space: ImageSpace
) -> JointIRLSFourier:
    # Build IRLSSolver estimator
    params["solver_params"]["space"] = ImageSpace.FOURIER_COMPLEX
    solver = _build_m_estimator(params, image_batch, ImageSpace.FOURIER_COMPLEX)
    return JointIRLSFourier(solver)


def _build_flattening_fourier(
    params: dict, image_batch: ImageBatch, space: ImageSpace
) -> FlatteningIRLSFourier:
    solver = _build_m_estimator(params, image_batch, ImageSpace.FOURIER_REAL)
    return FlatteningIRLSFourier(solver)


def _build_recursive_gmm(
    params: dict, image_batch: ImageBatch, space: ImageSpace
) -> RecursiveGMMEstimator:
    distance_func = get_distance_function(
        name=params["distance_function"],
        params=params.get("distance_params", {}),
        imgs=image_batch.select_space(space),
    )
    return RecursiveGMMEstimator(
        distance_function=distance_func,
        random_state=params.get("random_state", None),
        max_iter=params.get("max_iter", 10),
        tol=params.get("tol", 1e-3),
    )


def _build_admm(params: dict, image_batch: ImageBatch, space: ImageSpace) -> ADMMSolver:
    irls_real = build_estimator(params["real_estimator"], image_batch, ImageSpace.REAL)
    irls_fourier = build_estimator(
        params["fourier_estimator"], image_batch, ImageSpace.FOURIER_COMPLEX
    )
    return ADMMSolver(
        irls_real=irls_real,
        irls_fourier=irls_fourier,
        **params.get("solver_params", {}),
    )


def _build_masked_fourier(
    params: dict, image_batch: ImageBatch, space: ImageSpace
) -> MaskedIRLSFourier:
    solver = _build_m_estimator(params, image_batch, ImageSpace.FOURIER_COMPLEX)
    return MaskedIRLSFourier(solver=solver)


def _build_three_masks_fourier(
    params: dict, image_batch: ImageBatch, space: ImageSpace
) -> ThreeMasksFourier:
    solver = _build_masked_fourier(params, image_batch, ImageSpace.FOURIER_COMPLEX)
    return ThreeMasksFourier(
        low_cutoff=params["low_cutoff"],
        high_cutoff=params["high_cutoff"],
        solver=solver,
    )


def _build_fourier_gmm(params: dict, image_batch: ImageBatch, space: ImageSpace):
    lowpass_mask = create_lowpass_rfft_mask(
        image_shape=image_batch.real_shape,
        cutoff=LOWPASS_NORMALIZED_CUTOFF,
        unit="normalized",
    )
    std = image_batch.fourier_modulus_std()[lowpass_mask]
    delta = params.pop("delta", DEFAULT_FOURIER_GMM_DELTA)

    def distance_function(images, reference):
        weights = norm_smooth_redescending_weights(
            images[:, lowpass_mask], reference[lowpass_mask], std=std, delta=delta
        )
        return -weights.reshape(images.shape[0], 1)

    solver = RecursiveGMMEstimator(
        distance_function=distance_function,
        random_state=params.get("random_state", None),
        max_iter=params.get("max_iter", 10),
        tol=params.get("tol", 1e-3),
    )

    return FourierGMM(solver=solver)


BUILDER_REGISTRY: dict[
    EstimatorTypeName, Callable[[dict, ImageBatch, ImageSpace], Estimator]
] = {
    "m_estimator": _build_m_estimator,
    "fourier_m_estimator": _build_fourier_m_estimator,
    "joint_fourier": _build_joint_fourier,
    "flattening_fourier": _build_flattening_fourier,
    "recursive_gmm": _build_recursive_gmm,
    "admm": _build_admm,
    "masked_fourier": _build_masked_fourier,
    "three_masks_fourier": _build_three_masks_fourier,
    "fourier_gmm": _build_fourier_gmm,
}


def build_estimator(
    method_cfg: dict,
    image_batch: ImageBatch,
    space: ImageSpace = ImageSpace.REAL,
) -> Estimator:
    """
    Factory function that reads the YAML config block and returns
    the instantiated Estimator object on the specified device.
    """
    estimator_type = method_cfg["type"]
    params = deepcopy(method_cfg.get("params", {}))

    if estimator_type not in BUILDER_REGISTRY:
        raise ValueError(
            f"Unknown estimator type: {estimator_type}. "
            f"Valid types are {get_args(EstimatorTypeName)}"
        )

    return BUILDER_REGISTRY[estimator_type](params, image_batch, space)
