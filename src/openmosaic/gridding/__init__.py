# Copyright (c) 2020 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Grid definitions and transforms along with regridding utils."""

import cartopy.crs as ccrs
from metpy.plots.mapping import CFProjection
import numpy as np
import pyproj
import xarray as xr

from .vendored import map_gates_to_grid


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
    crs = CFProjection(cf_attrs).to_cartopy()
    if x0 is None or y0 is None:
        x0 = -(nx - 1) / 2 * dx
        y0 = -(ny - 1) / 2 * dy
    
    # Generate grid
    x = np.arange(nx) * dx + x0
    y = np.arange(ny) * dy + y0
    xx, yy = np.meshgrid(x, y)
    lonlats = ccrs.PlateCarree().transform_points(crs, xx, yy)
    lon = lonlats[..., 0]
    lat = lonlats[..., 1]

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
    

def generate_3d_radar_grid(
    radars,
    fields,
    subbatch,
    cf_attrs,
    weighting_function='gridrad',
    spacing=2e3,
    nz=24,
    dz=1e3,
    z0=1e3,
    **kwargs
):
    """Map radars onto a common 3D grid as defined by subbatch params and cf_attrs.
    
    Parameters
    ----------
    radars : iterable of pyart.core.Radar
        Collection of radars to mosaic
    fields : iterable of str
        Field labels to include in gridding process
    subbatch : pandas.Series or dict-like
        Must have labels x_min, x_max, y_min, y_max, and analysis_time
    cf_attrs : dict
        Projection attributes following the CF Conventions
    weighting_function : str, optional
        Py-ART weighting function option. Defaults to 'gridrad', OpenMosaic's custom
        implementation
    spacing : float, optional
        Horizontal grid spacing, defaults to 2000 (in units of projection space)
    nz : int, optional
        Number of vertical altitude grid levels, defaults to 24
    dz : float, optional
        Vertical altitude grid spacing, defaults to 1000 (in meters)
    z0 : float, optional
        Lowest altitude level, defaults to 1000 (in meters)
    **kwargs
        Any additional arguments passed to `map_gates_to_grid`

    Returns
    -------
    xarray.Dataset
        Dataset of fields with metadata following CF Conventions

    """
    crs_pyproj = pyproj.CRS.from_cf(cf_attrs)
    crs_cartopy = CFProjection(cf_attrs).to_cartopy()
    lon_origin, lat_origin = ccrs.PlateCarree().transform_point(0, 0, crs_cartopy)

    params = rounded_grid_params(subbatch, spacing)
    grids = map_gates_to_grid(
        radars,
        (nz, params['ny'], params['nx']),
        (
            (z0, z0 + (nz - 1) * dz),
            (params['y_min'], params['y_max']),
            (params['x_min'], params['x_max'])
        ),
        grid_origin=(lat_origin, lon_origin),
        grid_origin_alt=0,
        grid_projection=crs_pyproj.to_dict(),
        fields=fields,
        weighting_function=weighting_function,
        analysis_time=subbatch['analysis_time'],
        **kwargs
    )

    ds = generate_rectangular_grid(
        params['nx'],
        params['ny'],
        spacing,
        spacing,
        cf_attrs, 
        x0=params['x_min'],
        y0=params['y_min']
    )
    ds = ds.assign_coords(
        z=xr.DataArray(
            np.arange(nz) * dz + z0,
            dims=('z',),
            name='z',
            attrs={
                'standard_name': 'altitude',
                'units': 'meter',
                'positive': 'up'
            }
        ),
        time=xr.DataArray(
            subbatch['analysis_time'],
            name='time',
            attrs={
                'long_name': 'Time of radar analysis'
            }
        )
    )
    ds = ds.assign(
        {
            field: xr.Variable(
                ('z', 'y', 'x'),
                grids[field],
                {k: v for k, v in radars[0].fields[field].items() if k in ['units', 'standard_name', 'long_name']}
            )
            for field in fields
        }
    )

    return ds
