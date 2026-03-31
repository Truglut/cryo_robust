import torch
from typing import Callable, Tuple
from .base import Estimator
from .utils import weighted_average


@torch.no_grad()
def regularised_irls(
    y: torch.Tensor,
    x0: torch.Tensor,
    std: torch.Tensor,
    weights_function: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    max_iter: int = 50,
    tol: float = 1e-3,
    damping_coef: float = 0,
    min_weight: float | None = None,
    max_weight: float | None = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
    """
    Computes the Iteratively Reweighted Least Squares (IRLS) algorithm with
    optional weight capping and update damping for robust estimation.

    Parameters
    ----------
    y : torch.Tensor
        Input observations. Shape: (n, height, width).
    x0 : torch.Tensor
        Initial estimate of the solution. Shape: (height, width).
    std : torch.Tensor
        Per-pixel standard deviation or a scalar for global noise.
        Used by the weights_function to normalize residuals.
    weights_function : Callable
        Function expected to have the signature f(y, x, std) -> weights.
        Should return a tensor of the same shape as 'y'.
    max_iter : int, optional
        Maximum number of iterations to perform, by default 50.
    tol : float, optional
        Convergence threshold based on the L2 norm of the update, by default 1e-3.
    damping_coef : float, optional
        Momentum factor in [0, 1). Higher values increase stability but slow
        convergence, by default 0.
    min_weight : float | None, optional
        Lower bound for weights to prevent division by zero or singular
        solutions, by default None.
    max_weight : float | None, optional
        Upper bound for weights to limit the influence of specific
        observations, by default None.
    verbose : bool, optional
        If True, prints warnings regarding NaNs and convergence, by default False.

    Returns
    -------
    x : torch.Tensor
        The final estimated solution. Shape: (height, width).
    weights : torch.Tensor
        The weights calculated in the last iteration. Shape: (n, height, width).
    iterations : int
        The number of iterations actually performed.
    converged : bool
        True if the update norm fell below 'tol' before 'max_iter'.
    """
    x = x0
    converged = False
    weights = torch.ones_like(y)

    for i in range(max_iter):
        # Update Weights: Evaluate influence of current residuals
        weights = weights_function(y, x, std)

        if verbose and torch.isnan(weights).any():
            print(
                f"Warning [regularised_irls]: weights contain NaN at iteration {i + 1}"
            )

        # Regularisation: Weight capping
        # Avoids numerical instability or extreme outliers
        if min_weight is not None or max_weight is not None:
            weights = torch.clamp(weights, min=min_weight, max=max_weight)

        # Computing the weighted average (eps is added to denominator for numerical stability)
        x_next = weighted_average(y, weights, eps=1e-8)

        # Regularisation: Update damping (Momentum)
        x_next = (1 - damping_coef) * x_next + damping_coef * x

        if verbose and torch.isnan(x_next).any():
            print(
                f"Warning [regularised_irls]: estimate contains NaN at iteration {i + 1}"
            )

        # Convergence check: L2 norm of the step
        step_diff = torch.norm(x - x_next)
        if step_diff < tol:
            converged = True
            x = x_next
            break

        x = x_next

    return x, weights, i + 1, converged


class MEstimator(Estimator):
    def __init__(
        self,
        weight_function: Callable,
        x0: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        tol: float = 1e-3,
        max_iter: int = 100,
        damping_coef: float = 0.0,
        min_weight: float | None = None,
        max_weight: float | None = None,
        device: str = "cpu"
    ):
        super().__init__(device=device)
        self.weight_function = weight_function
        self.tol = tol
        self.max_iter = max_iter
        self.x0 = x0
        self.std = std
        self.damping_coef = damping_coef
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.converged = False

    def fit(self, images: torch.Tensor, verbose: bool = False):
        # Convert numpy arrays to pytorch tensor
        tensor_images = self._prepare_data(images)

        if self.x0 is None:
            x0 = tensor_images.mean(dim=0)
        else:
            x0 = self._prepare_data(self.x0)

        if self.std is None:
            std = tensor_images.std(dim=0)
        else:
            std = self._prepare_data(self.std)

        self.avg, self.final_weights, self.n_its, self.converged = regularised_irls(
            y=tensor_images,
            x0=x0,
            std=std,
            weights_function=self.weight_function,
            max_iter=self.max_iter,
            tol=self.tol,
            damping_coef=self.damping_coef,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
            verbose=verbose,
        )
