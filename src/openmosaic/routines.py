# Copyright (c) 2020 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Implementations of field calculation and mosaicing algorithms."""

import numpy as np
from scipy.ndimage import median_filter
import xarray as xr

from ._c_routines import sl3d_grid


def hello_world():
    """Test that tests and project configuration work."""
    return np.mean([41, 43])


def sl3d(reflectivity, z_melt, z_surface=0, dx=2e3, mask_below=20):
    """Implement the SL3D classification algorithm.

    Parameters
    ----------
    reflectivity: xr.DataArray
        Reflectivity. Must have altitude as increasing z coordinate, and be in (Z, Y, X)
        dimension order and in units of dBZ.
    z_melt: xr.DataArray
        Melting level altitude, in units of meters.
    z_surface: xr.DataArray or float or int, optional
        Altitude ASL of surface. Defaults to zero to force SL3D to use ASL rather than AGL
        heights.
    dx: float, optional
        Horizontal grid spacing, using to compute median filter shape
    mask_below: float, optional
        Set classification to none (0) if below a certain value. Defaults to 20. Set to 0
        to do nothing (since procedure assumes 0 and below are equivalent to missing data).

    Returns
    -------
    xr.DataArray
        SL3D Classification with integer encoding. 0: no classification, 1: convection,
        2: precip stratiform, 3: non-precip stratiform, 4: anvil, 5: updraft
    """
    # Median filter with 12 km radius
    radius = int(1.2e4 / dx)
    size = 2 * radius + 1
    x, y = np.mgrid[:size, :size]
    distance = np.sqrt((x - radius) ** 2 + (y - radius) ** 2)
    circle = distance <= radius
    filtered_refl = median_filter(
        reflectivity.data.astype("float32"),
        footprint=circle[None],
        mode="nearest"
    )

    return xr.DataArray(
        sl3d_grid(
            reflectivity.data.astype("float32"),
            reflectivity["z"].data.astype("float32"),
            np.atleast_2d(np.asarray(z_surface, dtype="float32")),
            z_melt.data.astype("float32"),
            filtered_refl,
            float(mask_below)
        ),
        coords={k: v for k, v in reflectivity.coords.items() if k != "z"},
        dims=("y", "x"),
        attrs={
            'long_name': 'SL3D Echo Classification',
            'description': (
                '0: no classification, 1: convection, 2: precip stratiform, '
                '3: non-precip stratiform, 4: anvil, 5: updraft'
            )
        }
    )


    