# Copyright (c) 2022 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Underlying numerical routines for DVAD-LLSD hybrid approach.

TODO:
- Allow separated rVd and moment calculation (e.g., rVd calculated on polar, then gridded,
  then moments)
- Docstrings
- Consider abstracting out kernel handling (rather than assuming only a circular kernel of
  configurable size)
- Need to include masking/missing data handling (i.e., force weight to zero)
"""

import numba
import numpy as np


# LLSD
@numba.njit()
def dvad_llsd_local(velocity_field, x_field, y_field, weights, x_center, y_center):
    terms = np.zeros((3,3))
    sum_w_rvd = 0.0
    sum_w_rvd_delA = 0.0
    sum_w_rvd_delB = 0.0
    sum_w_rvd_delC = 0.0
    sum_w_rvd_delD = 0.0

    for i in range(velocity_field.shape[1]):
        for j in range(velocity_field.shape[0]):
            vd = velocity_field[j, i]
            xp = x_field[j, i] - x_center
            yp = y_field[j, i] - y_center
            r = np.hypot(x_field[j, i], y_field[j, i])
            w = weights[j, i]
            
            delA = (xp * xp + x_center * xp + yp * yp + y_center * yp) * 0.5
            delB = (y_center * xp - x_center * yp) * 0.5

            # Matrix
            terms[0, 0] += w
            terms[0, 1] += w * delA
            terms[0, 2] += w * delB
            terms[1, 1] += w * delA * delA
            terms[1, 2] += w * delA * delB
            terms[2, 2] += w * delB * delB
            
            # Y vector
            w_rvd = r * vd * w
            sum_w_rvd += w_rvd
            sum_w_rvd_delA += w_rvd * delA
            sum_w_rvd_delB += w_rvd * delB
            
    # Matching terms of symmetric matrix
    terms[1, 0] = terms[0, 1]
    terms[2, 0] = terms[0, 2]
    terms[2, 1] = terms[1, 2]

    # Matrix solve
    Y = np.array([sum_w_rvd, sum_w_rvd_delA, sum_w_rvd_delB])
    X = np.linalg.solve(terms, Y)

    return X


@numba.njit()
def dvad_llsd_neighborhood_solve_polar(
    nray, ngate, vel_field, x_2d, y_2d, neighborhood_radius, max_rays, range_delta
):
    null_return = np.full(3, np.nan)
    # Center
    x_center = x_2d[nray, ngate]
    y_center = y_2d[nray, ngate]
    
    # Solve for gate offset
    m = int(neighborhood_radius / range_delta)
    i_min = ngate - (m // 2)
    i_max = ngate + (m // 2) + 1
    
    # March in azimuth to find bound
    within_j_bounds = True
    for j in range(1, max_rays // 2):
        if nray + j >= vel_field.shape[0]:
            within_j_bounds = False
            break
        neighborhood_distance = np.hypot(x_2d[nray + j] - x_center, y_2d[nray + j] - y_center)
        if np.all(neighborhood_distance > neighborhood_radius):
            # this is outer bound...if not, continue until next
            break
    if not within_j_bounds:
        return null_return
    j_min = nray - j + 1
    j_max = nray + j
    
    # secondary bound check
    if i_min < 0 or j_min < 0 or i_max > vel_field.shape[1] or j_max > vel_field.shape[0]:
        return null_return
        
    # Calculate weights from circle
    neighborhood_distance = np.hypot(x_2d[j_min:j_max, i_min:i_max] - x_center, y_2d[j_min:j_max, i_min:i_max] - y_center)
    weights = (neighborhood_distance <= neighborhood_radius) * 1

    # Calculate value
    return dvad_llsd_local(
        vel_field[j_min:j_max, i_min:i_max],
        x_2d[j_min:j_max, i_min:i_max],
        y_2d[j_min:j_max, i_min:i_max],
        weights,
        x_center,
        y_center
    )


@numba.njit()
def vorticity_llsd_dvad(
    vel_field, x_2d, y_2d, neighborhood_radius=1500, max_rays=51, range_delta=250
):
    out = np.full_like(vel_field, np.nan)
    for nray in range(1, out.shape[0] - 1):
        for ngate in range(1, out.shape[1] - 1):
            out[nray, ngate] = dvad_llsd_neighborhood_solve_polar(
                nray, ngate, vel_field, x_2d, y_2d, neighborhood_radius, max_rays, range_delta
            )[2]  # choose 1 for divergence, 2 for vorticity
    
    return out


@numba.njit()
def divergence_llsd_dvad(
    vel_field, x_2d, y_2d, neighborhood_radius=1500, max_rays=51, range_delta=250
):
    out = np.full_like(vel_field, np.nan)
    for nray in range(1, out.shape[0] - 1):
        for ngate in range(1, out.shape[1] - 1):
            out[nray, ngate] = dvad_llsd_neighborhood_solve_polar(
                nray, ngate, vel_field, x_2d, y_2d, neighborhood_radius, max_rays, range_delta
            )[1]  # choose 1 for divergence, 2 for vorticity
    
    return out