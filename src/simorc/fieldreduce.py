"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Proper Orthogonal Decomposition (POD) / SVD spatial field reduction.
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np
from .field import FieldLayout


class EFieldReductionCenter(Enum):
    """Centering policy for SVD snapshot decomposition."""

    none = "none"
    mean = "mean"


@dataclass(slots=True)
class FieldBasis:
    """Reduced spatial POD basis for field projection and reconstruction."""

    layout: FieldLayout
    mean: np.ndarray | None
    modes: np.ndarray  # shape: (num_dofs, num_modes)
    singular_values: np.ndarray
    num_modes: int
    reduction_center: EFieldReductionCenter

    def project(self, snapshots: np.ndarray) -> np.ndarray:
        """Project snapshot vectors onto the spatial modes.

        Parameters
        ----------
        snapshots : np.ndarray
            Matrix of flattened fields of shape (num_samples, num_dofs).

        Returns
        -------
        np.ndarray
            Modal coefficient matrix of shape (num_samples, num_modes).
        """
        if snapshots.ndim == 1:
            snapshots = snapshots.reshape(1, -1)
        centered = snapshots.copy()
        if self.mean is not None:
            centered -= self.mean
        return np.matmul(centered, self.modes)

    def reconstruct(self, coeffs: np.ndarray) -> np.ndarray:
        """Reconstruct flattened field vectors from modal coefficients.

        Parameters
        ----------
        coeffs : np.ndarray
            Modal coefficients of shape (num_samples, num_modes) or
            (num_modes,).

        Returns
        -------
        np.ndarray
            Reconstructed flattened vectors of shape (num_samples, num_dofs).
        """
        if coeffs.ndim == 1:
            coeffs = coeffs.reshape(1, -1)
        recon = np.matmul(coeffs, self.modes.T)
        if self.mean is not None:
            recon += self.mean
        return recon


@dataclass(slots=True)
class FieldReductionValidation:
    """Validation metrics for modal reduction across mode counts."""

    reconstruction_rmse: np.ndarray
    reconstruction_max_error: np.ndarray
    energy_fraction: np.ndarray


def calc_field_basis(
    snapshot_matrix: np.ndarray,
    layout: FieldLayout,
    num_modes: int | None = None,
    energy_fraction: float | None = None,
    center: EFieldReductionCenter = EFieldReductionCenter.mean,
) -> FieldBasis:
    """Calculate SVD/POD spatial modes from a snapshot matrix.

    Parameters
    ----------
    snapshot_matrix : np.ndarray
        Matrix of shape (num_samples, num_dofs).
    layout : FieldLayout
        Layout metadata for spatial points and components.
    num_modes : int | None, optional
        Fixed number of modes to retain, by default None.
    energy_fraction : float | None, optional
        Singular value energy fraction to retain (e.g. 0.999), by default None.
    center : EFieldReductionCenter, optional
        Whether to subtract mean field prior to SVD, by default mean.

    Returns
    -------
    FieldBasis
        Trained field basis object.
    """
    if snapshot_matrix.ndim == 1:
        snapshot_matrix = snapshot_matrix.reshape(1, -1)

    num_samples, num_dofs = snapshot_matrix.shape

    mean_vec = None
    work_mat = snapshot_matrix.copy()
    if center == EFieldReductionCenter.mean:
        mean_vec = np.mean(work_mat, axis=0, keepdims=True)
        work_mat -= mean_vec

    _, sing_vals, v_trans = np.linalg.svd(work_mat, full_matrices=False)
    full_modes = v_trans.T

    max_rank = len(sing_vals)
    if num_modes is not None:
        retained_modes = min(num_modes, max_rank)
    elif energy_fraction is not None:
        cum_energy = np.cumsum(sing_vals**2) / np.sum(sing_vals**2)
        retained_modes = int(np.searchsorted(cum_energy, energy_fraction)) + 1
        retained_modes = min(retained_modes, max_rank)
    else:
        retained_modes = max_rank

    retained_basis = full_modes[:, :retained_modes]

    return FieldBasis(
        layout=layout,
        mean=mean_vec,
        modes=retained_basis,
        singular_values=sing_vals,
        num_modes=retained_modes,
        reduction_center=center,
    )


def calc_field_mode_errors(
    snapshot_matrix: np.ndarray,
    basis: FieldBasis,
    max_modes: int | None = None,
) -> FieldReductionValidation:
    """Calculate reconstruction error as a function of retained mode count.

    Parameters
    ----------
    snapshot_matrix : np.ndarray
        Matrix of shape (num_samples, num_dofs).
    basis : FieldBasis
        Calculated full SVD basis.
    max_modes : int | None, optional
        Maximum mode count to evaluate, by default evaluates all modes.

    Returns
    -------
    FieldReductionValidation
        Error metrics across mode counts.
    """
    if max_modes is None:
        max_modes = basis.num_modes

    num_samples = snapshot_matrix.shape[0]
    total_energy = np.sum(basis.singular_values**2)

    rmse_list = []
    max_err_list = []
    energy_list = []

    for rank in range(1, max_modes + 1):
        temp_basis = FieldBasis(
            layout=basis.layout,
            mean=basis.mean,
            modes=basis.modes[:, :rank],
            singular_values=basis.singular_values[:rank],
            num_modes=rank,
            reduction_center=basis.reduction_center,
        )
        coeffs = temp_basis.project(snapshot_matrix)
        recon = temp_basis.reconstruct(coeffs)
        diff = snapshot_matrix - recon

        rmse = np.sqrt(np.mean(diff**2))
        max_err = np.max(np.abs(diff))
        energy = np.sum(basis.singular_values[:rank] ** 2) / total_energy

        rmse_list.append(rmse)
        max_err_list.append(max_err)
        energy_list.append(energy)

    return FieldReductionValidation(
        reconstruction_rmse=np.array(rmse_list, dtype=np.float64),
        reconstruction_max_error=np.array(max_err_list, dtype=np.float64),
        energy_fraction=np.array(energy_list, dtype=np.float64),
    )
