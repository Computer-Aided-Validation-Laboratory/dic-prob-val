"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Gaussian Process regression for scalar and reduced modal field outputs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.optimize import minimize
from .dataset import TrainingData
from .field import FieldLayout, restore_field_layout
from .fieldreduce import FieldBasis
from .param import ParamValues


@dataclass(slots=True)
class ConfigGaussianProcess:
    """Hyperparameter configuration for Gaussian Process regression."""

    kernel: str = "rbf"
    noise_level: float = 1e-6
    normalize_inputs: bool = True
    normalize_outputs: bool = True
    n_restarts: int = 3
    seed: int = 42


@dataclass(slots=True)
class SurrogateValidation:
    """Surrogate model performance validation metrics on test dataset."""

    output_names: tuple[str, ...]
    rmse: np.ndarray
    nrmse: np.ndarray
    max_abs_error: np.ndarray
    r2_score: np.ndarray


class ISurrogate(ABC):
    """Abstract surrogate model interface."""

    @abstractmethod
    def fit(self, data: TrainingData) -> "ISurrogate":
        """Fit surrogate model to training data."""
        ...

    @abstractmethod
    def predict(
        self, inputs: np.ndarray | ParamValues
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict output means and variances."""
        ...


def _calc_sq_dist(
    x1: np.ndarray, x2: np.ndarray, length_scale: np.ndarray
) -> np.ndarray:
    """Compute scaled squared Euclidean distance matrix."""
    s1 = x1 / length_scale
    s2 = x2 / length_scale
    dist_sq = (
        np.sum(s1**2, axis=1, keepdims=True)
        + np.sum(s2**2, axis=1, keepdims=True).T
        - 2.0 * np.matmul(s1, s2.T)
    )
    return np.maximum(dist_sq, 0.0)


@dataclass(slots=True)
class SingleOutputGP:
    """Self-contained Gaussian Process for a single scalar output."""

    config: ConfigGaussianProcess
    x_train: np.ndarray | None = None
    y_train: np.ndarray | None = None
    x_mean: np.ndarray | None = None
    x_std: np.ndarray | None = None
    y_mean: float = 0.0
    y_std: float = 1.0
    length_scale: np.ndarray | None = None
    sigma_f2: float = 1.0
    k_inv: np.ndarray | None = None
    alpha: np.ndarray | None = None

    def fit(self, x_in: np.ndarray, y_in: np.ndarray) -> "SingleOutputGP":
        """Fit GP hyperparameters via negative log marginal likelihood."""
        num_samples, dim = x_in.shape
        self.x_mean = (
            np.mean(x_in, axis=0)
            if self.config.normalize_inputs
            else np.zeros(dim)
        )
        self.x_std = (
            np.std(x_in, axis=0) + 1e-12
            if self.config.normalize_inputs
            else np.ones(dim)
        )
        self.y_mean = (
            float(np.mean(y_in)) if self.config.normalize_outputs else 0.0
        )
        self.y_std = (
            float(np.std(y_in) + 1e-12)
            if self.config.normalize_outputs
            else 1.0
        )

        x_norm = (x_in - self.x_mean) / self.x_std
        y_norm = (y_in - self.y_mean) / self.y_std
        self.x_train = x_norm
        self.y_train = y_norm

        def nll(theta: np.ndarray) -> float:
            ls = np.exp(theta[:dim])
            sig_f2 = np.exp(theta[dim])
            dist_sq = _calc_sq_dist(x_norm, x_norm, ls)
            cov = sig_f2 * np.exp(-0.5 * dist_sq) + (
                self.config.noise_level + 1e-8
            ) * np.eye(num_samples)
            try:
                l_chol = np.linalg.cholesky(cov)
                al = np.linalg.solve(l_chol.T, np.linalg.solve(l_chol, y_norm))
                log_det = 2.0 * np.sum(np.log(np.diag(l_chol)))
                return 0.5 * float(np.dot(y_norm, al)) + 0.5 * log_det
            except np.linalg.LinAlgError:
                return 1e20

        rng = np.random.default_rng(self.config.seed)
        best_nll = 1e25
        best_theta = np.zeros(dim + 1)
        opt_bounds = [(-4.0, 4.0)] * dim + [(-3.0, 3.0)]

        for _ in range(self.config.n_restarts):
            init_theta = np.concatenate(
                [rng.uniform(-0.5, 0.5, size=dim), [0.0]]
            )
            opt_res = minimize(
                nll, init_theta, method="L-BFGS-B", bounds=opt_bounds
            )
            if opt_res.fun < best_nll:
                best_nll = opt_res.fun
                best_theta = opt_res.x

        self.length_scale = np.exp(best_theta[:dim])
        self.sigma_f2 = float(np.exp(best_theta[dim]))

        dist_sq = _calc_sq_dist(x_norm, x_norm, self.length_scale)
        cov = self.sigma_f2 * np.exp(-0.5 * dist_sq) + (
            self.config.noise_level + 1e-8
        ) * np.eye(num_samples)
        l_chol = np.linalg.cholesky(cov)
        eye_mat = np.eye(num_samples)
        self.k_inv = np.linalg.solve(
            l_chol.T, np.linalg.solve(l_chol, eye_mat)
        )
        self.alpha = np.dot(self.k_inv, y_norm)
        return self

    def predict(self, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict normalized mean and variance at query points."""
        if x_query.ndim == 1:
            x_query = x_query.reshape(1, -1)
        x_norm = (x_query - self.x_mean) / self.x_std

        dist_sq_star = _calc_sq_dist(x_norm, self.x_train, self.length_scale)
        k_star = self.sigma_f2 * np.exp(-0.5 * dist_sq_star)

        y_mean_norm = np.dot(k_star, self.alpha)
        y_pred = self.y_mean + self.y_std * y_mean_norm

        v_mat = np.dot(k_star, self.k_inv)
        y_var_norm = np.maximum(
            self.sigma_f2 - np.sum(v_mat * k_star, axis=1), 0.0
        )
        y_var = (self.y_std**2) * y_var_norm
        return y_pred, y_var


@dataclass(slots=True)
class SurrogateGaussianProcess(ISurrogate):
    """Multi-output Gaussian Process surrogate model."""

    config: ConfigGaussianProcess
    param_names: tuple[str, ...]
    output_names: tuple[str, ...]
    models: tuple[SingleOutputGP, ...] = ()

    def fit(self, data: TrainingData) -> "SurrogateGaussianProcess":
        """Fit independent GP for each output dimension."""
        num_outputs = data.get_num_outputs()
        gp_models = []
        for jj in range(num_outputs):
            gp = SingleOutputGP(config=self.config)
            gp.fit(data.inputs, data.outputs[:, jj])
            gp_models.append(gp)

        return SurrogateGaussianProcess(
            config=self.config,
            param_names=data.param_names,
            output_names=data.output_names,
            models=tuple(gp_models),
        )

    def predict(
        self, inputs: np.ndarray | ParamValues
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance for all output dimensions."""
        x_arr = inputs.values if isinstance(inputs, ParamValues) else inputs
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        num_samples = x_arr.shape[0]
        num_outputs = len(self.output_names)

        means = np.zeros((num_samples, num_outputs), dtype=np.float64)
        variances = np.zeros((num_samples, num_outputs), dtype=np.float64)

        for jj, gp in enumerate(self.models):
            m, v = gp.predict(x_arr)
            means[:, jj] = m
            variances[:, jj] = v

        return means, variances

    def validate(self, test_data: TrainingData) -> SurrogateValidation:
        """Calculate validation metrics against test dataset."""
        pred_means, _ = self.predict(test_data.inputs)
        y_true = test_data.outputs
        diff = y_true - pred_means

        rmse = np.sqrt(np.mean(diff**2, axis=0))
        denom = np.max(y_true, axis=0) - np.min(y_true, axis=0) + 1e-12
        nrmse = rmse / denom
        max_abs = np.max(np.abs(diff), axis=0)

        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0) + 1e-12
        ss_res = np.sum(diff**2, axis=0)
        r2 = 1.0 - (ss_res / ss_tot)

        return SurrogateValidation(
            output_names=self.output_names,
            rmse=rmse,
            nrmse=nrmse,
            max_abs_error=max_abs,
            r2_score=r2,
        )


@dataclass(slots=True)
class SurrogateField(ISurrogate):
    """Modal-decomposed spatial field surrogate model."""

    basis: FieldBasis
    modal_surrogate: SurrogateGaussianProcess

    def fit(self, data: TrainingData) -> "SurrogateField":
        """Fit modal surrogate model."""
        trained_modal = self.modal_surrogate.fit(data)
        return SurrogateField(basis=self.basis, modal_surrogate=trained_modal)

    def predict(
        self, inputs: np.ndarray | ParamValues
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict spatial field values and variances at query points."""
        modal_means, modal_vars = self.modal_surrogate.predict(inputs)
        field_recon = self.basis.reconstruct(modal_means)

        field_vars = np.matmul(modal_vars, (self.basis.modes.T) ** 2)
        return field_recon, field_vars


def build_surrogate(
    data: TrainingData,
    config: ConfigGaussianProcess | None = None,
) -> SurrogateGaussianProcess:
    """Build and fit a Gaussian Process surrogate model."""
    if config is None:
        config = ConfigGaussianProcess()
    model = SurrogateGaussianProcess(
        config=config,
        param_names=data.param_names,
        output_names=data.output_names,
    )
    return model.fit(data)


def build_field_surrogate(
    data_inputs: np.ndarray | ParamValues,
    snapshots: np.ndarray,
    basis: FieldBasis,
    config: ConfigGaussianProcess | None = None,
) -> SurrogateField:
    """Build and fit a spatial field POD-GP surrogate model."""
    if config is None:
        config = ConfigGaussianProcess()

    x_mat = (
        data_inputs.values
        if isinstance(data_inputs, ParamValues)
        else data_inputs
    )
    modal_coeffs = basis.project(snapshots)
    mode_names = tuple(f"mode_{ii}" for ii in range(basis.num_modes))

    param_names = (
        data_inputs.names
        if isinstance(data_inputs, ParamValues)
        else tuple(f"param_{ii}" for ii in range(x_mat.shape[1]))
    )

    modal_data = TrainingData(
        param_names=param_names,
        output_names=mode_names,
        inputs=x_mat,
        outputs=modal_coeffs,
    )

    modal_gp = build_surrogate(modal_data, config=config)
    return SurrogateField(basis=basis, modal_surrogate=modal_gp)
