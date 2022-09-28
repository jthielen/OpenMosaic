# Copyright (c) 2022 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Core regridding routines and their mid-level wrappers."""

from numba import boolean, float32, njit, uint8
import numpy as np


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
