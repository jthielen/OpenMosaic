# Copyright (c) 2022 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Operations related to polar and rectangular grid coordinates."""

from time import sleep
from pathlib import Path
from joblib import dump, load

import numpy as np
import pyproj
from sklearn.linear_model import ElasticNet


def _get_regression_paths(site_id, cache_dir):
    regression_path = Path(cache_dir) / "./regressions"
    regression_path.mkdir(parents=True, exist_ok=True)
    return (
        regression_path / f"{site_id}_reg_x.joblib",
        regression_path / f"{site_id}_reg_y.joblib"
    )


def _get_quadratic_terms(x_input, y_input):
    x_coord_term = x_input.astype('float32').ravel()
    y_coord_term = y_input.astype('float32').ravel()
    return np.stack([
        x_coord_term,
        y_coord_term,
        x_coord_term**2,
        y_coord_term**2,
        x_coord_term * y_coord_term
    ], axis=1)


def transform_radar_coords_to_grid_coords(
    gate_radar_x,
    gate_radar_y,
    x_coordinate_regression,
    y_coordinate_regression
):
    """Map radar x/y to grid x/y using a quadratic model of projection transform.
    
    Uses a sklearn LinearModel regression (ElasticNet provided by
    generate_radar_coordinate_regressions recommended) over 2D quadratic terms to transform
    coordinates from radar-centered azimuthal equidistant space to a common grid.

    When used with an ElasticNet regression trained on relatively-prime-strided subsets of
    coordinate data from pyproj.Transformer, this technique gives more than an order of
    magnitude speed up over direct use of the pyproj.Transformer, with errors less than 100 m
    within 300 km of the radar site. It is also faster and significantly more accurate than
    using pyproj.Proj from Py-ART's calculated longitudes and latitudes.

    Parameters
    ----------
    gate_radar_x : numpy.ndarray
        Gate center x coordinate values (in meters) in radar-centered azimuthal equidistant
        projected grid
    gate_radar_y : numpy.ndarray
        Gate center y coordinate values (in meters) in radar-centered azimuthal equidistant
        projected grid
    x_coordinate_regression : sklearn.linear_model._base.LinearModel
        2D quadratic regression (term order: X, Y, X**2, Y**2, X * Y) to transform
        radar-centered projection x coordinates to grid x coordinates
    y_coordinate_regression : sklearn.linear_model._base.LinearModel
        2D quadratic regression (term order: X, Y, X**2, Y**2, X * Y) to transform
        radar-centered projection x coordinates to grid x coordinates

    Returns
    -------
    numpy.ndarray
        Grid coordinates of gate centers, with x and y stacked along first array dimension
    """
    coord_terms = _get_quadratic_terms(gate_radar_x, gate_radar_y)
    return np.stack(
        [
            regression.predict(coord_terms).reshape(target)
            for regression, target in (
                (x_coordinate_regression, gate_radar_x),
                (y_coordinate_regression, gate_radar_y)
            )
        ],
        axis=0
    )


def transform_radar_coords_to_grid_coords_direct(
    gate_radar_x,
    gate_radar_y,
    radar_crs_cf_attrs,
    target_crs_cf_attrs
):
    """Map radar x/y to grid x/y using a direct pyproj projection transform.

    For highest accuracy of coordinate values on target grid, use this function. In practice,
    the quadratic-regression-based transform_radar_coords_to_grid_coords will give a
    sufficiently accurate result with at least an order of magnitude less computational cost,
    and so, this direct function is only used to train the regressions.

    Parameters
    ----------
    gate_radar_x : numpy.ndarray
        Gate center x coordinate values (in meters) in radar-centered azimuthal equidistant
        projected grid
    gate_radar_y : numpy.ndarray
        Gate center y coordinate values (in meters) in radar-centered azimuthal equidistant
        projected grid
    radar_crs_cf_attrs : dict
        Dictionary of projection attributes for radar-centered grid mapping in accord with CF
        conventions
    target_crs_cf_attrs : dict
        Dictionary of projection attributes for the output grid's grid mapping in accord with
        CF conventions

    Returns
    -------
    numpy.ndarray
        Grid coordinates of gate centers, with x and y stacked along first array dimension
    """
    radar_crs = pyproj.CRS.from_cf(radar_crs_cf_attrs)
    target_crs = pyproj.CRS.from_cf(target_crs_cf_attrs)
    transformer = pyproj.Transformer.from_crs(radar_crs, target_crs)
    grid_x, grid_y = transformer.transform(gate_radar_x, gate_radar_y)
    return np.stack([grid_x, grid_y], axis=0)


def load_coord_transform_regressions(site_id, wait_for_cache=None, cache_dir='/tmp/'):
    """Attempt to load cached radar coordinate transform regressions.
    
    Parameters
    ----------
    site_id : str
        ID of radar site
    wait_for_cache : int, optional
        If specified as a positive integer, waits for specified number of seconds if dumped
        regression model not initially found (useful in parallelized context to avoid multiple
        workers retraining the same coordinate regression model, which can be a slow
        operation).
    cache_dir : str, optional
        Parent directory in which dumped regression models have been stored in the
        "regressions" subdirectory.

    Returns
    -------
    tuple of sklearn.linear_model.ElasticNet
        The x and y coordinate transform models, respectively
    """
    x_regression_path, y_regression_path = _get_regression_paths(site_id, cache_dir)

    x_clf = None
    y_clf = None
    if x_regression_path.exists() and y_regression_path.exists():
        x_clf = load(x_regression_path)
        y_clf = load(y_regression_path)
    elif wait_for_cache:
        # Wait specified count of seconds for cached version
        sleep(wait_for_cache)
        if x_regression_path.exists() and y_regression_path.exists():
            x_clf = load(x_regression_path)
            y_clf = load(y_regression_path)

    if x_clf is not None and y_clf is not None:
        return x_clf, y_clf
    else:
        raise FileNotFoundError(
            f"Unable to find regression models at paths {x_regression_path} and "
            f"{y_regression_path}"
        )


def create_coord_transform_regressions(
    gate_radar_x,
    gate_radar_y,
    site_id,
    radar_crs_cf_attrs,
    target_crs_cf_attrs,
    cache_dir='/tmp/',
    stride=11
):
    """Train 2D ElasticNet for radar to grid coordinate transform.
    
    Parameters
    ----------
    gate_radar_x : numpy.ndarray
        Gate center x coordinate values (in meters) in radar-centered azimuthal equidistant
        projected grid
    gate_radar_y : numpy.ndarray
        Gate center y coordinate values (in meters) in radar-centered azimuthal equidistant
        projected grid
    site_id : str
        ID of radar site. Must correspond one-to-one with a single location, as this is used
        as the key for loading the saved regression model later.
    radar_crs_cf_attrs : dict
        Dictionary of projection attributes for radar-centered grid mapping in accord with CF
        conventions
    target_crs_cf_attrs : dict
        Dictionary of projection attributes for the output grid's grid mapping in accord with
        CF conventions
    cache_dir : str, optional
        Parent directory in which dumped regression models will be stored in the
        "regressions" subdirectory.
    stride : int, optional
        Stride parameter for subsetting input data on which to train. Defaults to 11, which
        gives optimal results for 360 total rays of radar data. In general, this should be set
        to an integer relatively-prime to the number of rays where stride/nrays ~ 0.1 (a 10%
        subset). If stride is a divisor of the number of rays and stride >> 1, poor results
        may occur.

    Returns
    -------
    tuple of sklearn.linear_mode.ElasticNet
        The x and y coordinate transform models, respectively. These are also cached.
    """
    x_regression_path, y_regression_path = _get_regression_paths(site_id, cache_dir)
    coord_terms = _get_quadratic_terms(gate_radar_x, gate_radar_y)[slice(None, None, stride)]

    # Transform coordinates
    grid_x, grid_y = transform_radar_coords_to_grid_coords_direct(
        coord_terms[:, 0],
        coord_terms[:, 1],
        radar_crs_cf_attrs,
        target_crs_cf_attrs
    )

    # Train
    x_clf = ElasticNet().fit(coord_terms, grid_x)
    y_clf = ElasticNet().fit(coord_terms, grid_y)

    # Cache and return
    dump(x_clf, x_regression_path)
    dump(y_clf, y_regression_path)
    return x_clf, y_clf
