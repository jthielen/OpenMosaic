# Copyright (c) 2020 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Grid definitions and transforms along with regridding utils."""

import numpy as np
import pyproj
import xarray as xr


def rounded_grid_params(subbatch, spacing):
    """Return rounded/inclusive grid bounds and horizontal grid point counts given spacing.
    
    Parameters
    ----------
    subbatch : pandas.Series or dict-like
        Must have fields x_min, x_max, y_min, and y_max
    spacing : float
        Grid point spacing in same units as subbatch bounds

    Returns
    -------
    dict
        Dictionary containing rounded x_min, x_max, y_min, and y_max, as well as nx and ny
    """
    x_min = np.floor(subbatch['x_min'] / spacing) * spacing
    x_max = np.ceil(subbatch['x_max'] / spacing) * spacing
    y_min = np.floor(subbatch['y_min'] / spacing) * spacing
    y_max = np.ceil(subbatch['y_max'] / spacing) * spacing
    return {
        'nx': int((x_max - x_min) / spacing + 1),
        'ny': int((y_max - y_min) / spacing + 1),
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
    }


def generate_rectangular_grid(nx, ny, dx, dy, cf_attrs, x0=None, y0=None):
    """Generate an xarray Dataset representing a regular horizontal grid in projected space.

    Parameters
    ----------
    nx : int
        Number of grid points in x direction
    ny : int
        Numver of grid points in y direction
    dx : float
        Grid spacing in projected space in x direction
    dy : float
        Grid spacing in projected space in y direction
    cf_attrs : dict
        Dictionary of attributes describing the projection following the CF Conventions
    x0 : float, optional
        x coordinate of lower-left corner of grid. If None, defaults to centering the grid on
        the projection origin.
    y0 : float, optional
        y coordinate of lower-left corner of grid. If None, defaults to centering the grid on
        the projection origin.
    
    Returns
    -------
    xarray.Dataset
        Dataset describing grid, following CF Conventions
    """
    proj = pyproj.Proj(pyproj.CRS.from_cf(cf_attrs))
    if x0 is None or y0 is None:
        x0 = -(nx - 1) / 2 * dx
        y0 = -(ny - 1) / 2 * dy
    
    # Generate grid
    x = np.arange(nx) * dx + x0
    y = np.arange(ny) * dy + y0
    xx, yy = np.meshgrid(x, y)
    lon, lat = proj(xx, yy, inverse=True)

    # Create Dataset
    ds = xr.Dataset(
        coords={
            'lon': (['y', 'x'], lon),
            'lat': (['y', 'x'], lat),
            'y': y,
            'x': x
        }
    )
    ds['lon'].attrs = {
        'standard_name': 'longitude',
        'units': 'degrees_east'
    }
    ds['lat'].attrs = {
        'standard_name': 'latitude',
        'units': 'degrees_north'
    }
    ds['y'].attrs = {
        'standard_name': 'projection_y_coordinate',
        'units': 'meter'
    }
    ds['x'].attrs = {
        'standard_name': 'projection_x_coordinate',
        'units': 'meter'
    }
    ds['projection'] = xr.DataArray(np.array(0), attrs=cf_attrs)

    return ds
