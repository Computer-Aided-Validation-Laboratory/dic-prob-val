"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Spatial field layout, flattening, restoration, and common-grid standardisation.
"""

from dataclasses import dataclass
from typing import Union
import numpy as np
from scipy.interpolate import griddata
from .result import EFieldLocation, ResultField


@dataclass(slots=True)
class FieldLayout:
    """Explicit, reversible description of flattened spatial field data."""

    coords: np.ndarray
    components: tuple[str, ...]
    original_shape: tuple[int, ...]
    location: EFieldLocation = EFieldLocation.node

    def get_num_points(self) -> int:
        """Get number of spatial points."""
        return self.coords.shape[0]

    def get_num_components(self) -> int:
        """Get number of field components."""
        return len(self.components)

    def get_total_dofs(self) -> int:
        """Get total degrees of freedom (points * components)."""
        return self.get_num_points() * self.get_num_components()


def build_grid_layout(
    grid_shape: tuple[int, int],
    bounds: tuple[tuple[float, float], tuple[float, float]],
    components: tuple[str, ...] = ("val",),
    location: EFieldLocation = EFieldLocation.node,
) -> FieldLayout:
    """Construct a regular 2D grid FieldLayout.

    Parameters
    ----------
    grid_shape : tuple[int, int]
        Grid dimensions (num_y, num_x).
    bounds : tuple[tuple[float, float], tuple[float, float]]
        Spatial bounds ((x_min, x_max), (y_min, y_max)).
    components : tuple[str, ...], optional
        Component names, by default ('val',).
    location : EFieldLocation, optional
        Spatial location type, by default EFieldLocation.node.

    Returns
    -------
    FieldLayout
        Constructed regular 2D grid layout.
    """
    ny, nx = grid_shape
    (xmin, xmax), (ymin, ymax) = bounds
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    grid_x, grid_y = np.meshgrid(xs, ys)
    coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    return FieldLayout(
        coords=coords,
        components=components,
        original_shape=(ny, nx),
        location=location,
    )


def flatten_field(field_values: np.ndarray) -> np.ndarray:
    """Flatten field array into a 1D vector deterministically.

    Parameters
    ----------
    field_values : np.ndarray
        Array of shape (num_points, num_components) or (num_points,).

    Returns
    -------
    np.ndarray
        Flattened 1D array of length (num_points * num_components).
    """
    if field_values.ndim == 1:
        return field_values.copy()
    return field_values.reshape(-1, order="C")


def restore_field_layout(
    flat_vector: np.ndarray,
    layout: FieldLayout,
) -> np.ndarray:
    """Restore flattened vector into original spatial field layout.

    Parameters
    ----------
    flat_vector : np.ndarray
        1D flattened field array or 2D batch of shape (num_samples, dofs).
    layout : FieldLayout
        Target spatial field layout.

    Returns
    -------
    np.ndarray
        Restored array with spatial and component structure.
    """
    num_pts = layout.get_num_points()
    num_comp = layout.get_num_components()

    if flat_vector.ndim == 1:
        if num_comp == 1 and len(layout.original_shape) > 1:
            return flat_vector.reshape(layout.original_shape, order="C")
        return flat_vector.reshape((num_pts, num_comp), order="C")

    num_samples = flat_vector.shape[0]
    if num_comp == 1 and len(layout.original_shape) > 1:
        target_shape = (num_samples,) + layout.original_shape
        return flat_vector.reshape(target_shape, order="C")
    return flat_vector.reshape((num_samples, num_pts, num_comp), order="C")


def standardise_field_grid(
    field: ResultField,
    target: Union[np.ndarray, FieldLayout],
    method: str = "linear",
) -> ResultField:
    """Interpolate irregular mesh field data onto a standard common grid.

    Parameters
    ----------
    field : ResultField
        Source field containing native mesh coordinates and values.
    target : np.ndarray | FieldLayout
        Target coordinates matrix or FieldLayout object.
    method : str, optional
        Interpolation method ('linear', 'nearest', 'cubic'), default 'linear'.

    Returns
    -------
    ResultField
        Standardised field defined on target grid coordinates.
    """
    target_grid_coords = (
        target.coords if isinstance(target, FieldLayout) else target
    )
    src_coords = field.coords
    src_vals = field.values
    dim = src_coords.shape[1]

    if dim == 2 or target_grid_coords.shape[1] == 2:
        src_xy = src_coords[:, :2]
        target_xy = target_grid_coords[:, :2]
        interp_vals = griddata(
            src_xy, src_vals, target_xy, method=method, fill_value=0.0
        )
    else:
        interp_vals = griddata(
            src_coords,
            src_vals,
            target_grid_coords,
            method=method,
            fill_value=0.0,
        )

    if interp_vals.ndim == 1:
        interp_vals = interp_vals.reshape(-1, 1)

    return ResultField(
        name=field.name,
        values=interp_vals,
        coords=target_grid_coords.copy(),
        components=field.components,
        location=field.location,
    )
