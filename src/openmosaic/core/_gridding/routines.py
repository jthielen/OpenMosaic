# Copyright (c) 2022 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Core regridding routines and their mid-level wrappers."""

from time import sleep
from pathlib import Path
from joblib import dump, load

from numba import boolean, float32, njit, uint8
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


##########################################################################
## TODO REMOVE THIS DIVIDER                                             ##
##########################################################################
## Working on bottom-up re-implementation of subgrid mapping with numba ##
## ...fill in useful high-level stuff as we go...                       ##
##########################################################################

L2 = 2.25e10  # GridRad Length Scale Squared [m^2]
TAU2 = 2.25e4  # GridRad Time Scale Squared [s^2]
R_CUTOFF = 3.0e5  # GridRad Range Cutoff [m]
T_CUTOFF = 228.0  # GridRad Timedelta Cutoff [s]
DEPTH_BINS_CUTOFF = 3  # GridRad Depth Cutoff, given as the maximum number of bins in the
                       # vertical that a gate volume can contribute to


@njit(
    float32[:,:,:,:,:](
        float32[:,:,:],
        uint8[:,:,:],
        float32[:,:],
        float32[:,:],
        float32[:,:],
        float32[:],
        float32[:],
        float32[:],
        float32[:],
        float32[:],
        float32,
        boolean
    ),
    parallel=False,
    nogil=True
)
def map_radar_gates_to_subgrid_sum_gridrad_method(
    field_data,
    field_mask,
    gate_z_coord,
    gate_y_coord,
    gate_x_coord,
    gate_range,
    gate_timedelta,
    subgrid_z_coord,
    subgrid_y_coord,
    subgrid_x_coord,
    beam_width_radians=0.016580628,
    include_echo_count=False
):
    """Transform polar single-volume data to sums on a bounded subgrid, a la GridRad.
    
    Parameters
    ----------
    field_data : numpy.ndarray
        Shape: (nrays, ngates, nfields) TODO desc
    field_mask : numpy.ndarray
        Shape: (nrays, ngates, nfields) TODO desc
    gate_z_coord : numpy.ndarray
        Shape: (nrays, ngates) TODO desc must be strictly increasing
    gate_y_coord : numpy.ndarray
        Shape: (nrays, ngates) TODO desc must be strictly increasing
    gate_x_coord : numpy.ndarray
        Shape: (nrays, ngates) TODO desc must be strictly increasing
    gate_range : numpy.ndarray
        Shape: (ngates,) TODO desc
    gate_timedelta : numpy.ndarray
        Shape: (nrays,) TODO desc
    subgrid_z_coord : numpy.ndarray
        Shape: (nz,)
    subgrid_y_coord : numpy.ndarray
        Shape: (ny,)
    subgrid_x_coord : numpy.ndarray
        Shape: (nx,)
    beam_width_radians : float32, optional
        Angular beamwidth used to calculate beam contributing depth. Defaults to 0.95 deg, but
        converted to radians.
    include_echo_count : bool, optional
        Include an extra component counting the number of unique scan volumes contributing to
        that output grid volume

    Returns
    -------
    numpy.ndarray
        Shape: (ncomponents, nz, ny, nx, nfields) TODO desc...component dimension corresponds
        to (sum_of_weighted_values, sum_of_weights) when include_echo_counts is False
        (default), and (sum_of_weighted_values, sum_of_weights, N_echo) when True.

    Notes
    -----
    Developed from Steps 2, 3, and initial part of 4 of the GridRad v4.2 procedure described
    in http://gridrad.org/pdf/GridRad-v4.2-Algorithm-Description.pdf, Section 3.2
    """
    # Set array shapes
    nrays, ngates, nfields = field_data.shape
    nz, = subgrid_z_coord.shape
    ny, = subgrid_y_coord.shape
    nx, = subgrid_x_coord.shape

    # Set check values (based on inferring grid walls, i.e. extrapolation)
    z_min = 1.5 * subgrid_z_coord[0] - 0.5 * subgrid_z_coord[1]
    z_max = 1.5 * subgrid_z_coord[-1] - 0.5 * subgrid_z_coord[-2]
    y_min = 1.5 * subgrid_y_coord[0] - 0.5 * subgrid_y_coord[1]
    y_max = 1.5 * subgrid_y_coord[-1] - 0.5 * subgrid_y_coord[-2]
    x_min = 1.5 * subgrid_x_coord[0] - 0.5 * subgrid_x_coord[1]
    x_max = 1.5 * subgrid_x_coord[-1] - 0.5 * subgrid_x_coord[-2]

    # Output array
    ncomponents = 3 if include_echo_count else 2
    subgrid_sums = np.zeros((ncomponents, nz, ny, nx, nfields), dtype='float32')

    # TODO should we check other variables? sanity check coord increasing?

    # Loop over polar radar volumes
    for nray in range(nrays):
        t = gate_timedelta[nray]
        for ngate in range(ngates):
            # Scan volume parameters
            x = gate_x_coord[nray, ngate]
            y = gate_y_coord[nray, ngate]
            z = gate_z_coord[nray, ngate]
            r = gate_range[ngate]
            beam_depth_radius = beam_width_radians * r / 2
            values_by_field = field_data[nray, ngate]
            masks_by_field = field_mask[nray, ngate]

            # If fully masked or exceeding cutoff, can skip
            if (
                masks_by_field.all()
                or r > R_CUTOFF
                or t > T_CUTOFF
                or z > z_max
                or z < z_min
                or y > y_max
                or y < y_min
                or x > x_max
                or x < x_min
            ):
                continue

            # Calculate weight (which is altitude invariant)
            weight = np.exp(-r**2/L2) * np.exp(-t**2/TAU2)

            # Otherwise, search for output column to which to map this radar volume (nearest
            # neighbor by minimizing x and y differences)
            yi = np.abs(subgrid_y_coord - y).argmin()
            xi = np.abs(subgrid_x_coord - x).argmin()

            # Determine altitude bins in column to which to contribute
            zi_top = np.abs(subgrid_z_coord - (z + beam_depth_radius)).argmin()
            zi_bottom = np.abs(subgrid_z_coord - (z - beam_depth_radius)).argmin()
            if zi_top - zi_bottom + 1 > DEPTH_BINS_CUTOFF:
                # Single volume gate contributing to too much depth, need to limit
                trim_count = zi_top - zi_bottom + 1 - DEPTH_BINS_CUTOFF
                half_trim_count = trim_count // 2
                if np.mod(trim_count, 2) == 0:
                    # Trim evenly
                    zi_top -= half_trim_count
                    zi_bottom += half_trim_count
                elif subgrid_z_coord[np.abs(subgrid_z_coord - z).argmin()] - z < 0:
                    # Gate volume center above nearest vertical center, trim extra from bottom
                    zi_top -= half_trim_count
                    zi_bottom += half_trim_count + 1
                else:
                    # Gate volume center at or below nearest vertical center, trim extra from
                    # top
                    zi_top -= half_trim_count + 1
                    zi_bottom += half_trim_count

            # Update the values!
            for nfield in range(nfields):
                if not masks_by_field[nfield]:
                    # Weighted value goes in first, weight alone second
                    subgrid_sums[0, zi_bottom:(zi_top + 1), yi, xi, nfield] += (
                        weight * values_by_field[nfield]
                    )
                    subgrid_sums[1, zi_bottom:(zi_top + 1), yi, xi, nfield] += weight
                    if include_echo_count:
                        subgrid_sums[2, zi_bottom:(zi_top + 1), yi, xi, nfield] += 1

    # Return the combined sum array
    return subgrid_sums







