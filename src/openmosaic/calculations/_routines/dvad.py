# Copyright (c) 2022 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Underlying numerical routines for DVAD-LLSD hybrid approach.

TODO: refactor to share with LLSD approach(es) as well.
      also docstrings
"""

import numba
import numpy as np


# LLSD
@numba.njit()
def dvad_vorticity_local(velocity_field, x_field, y_field, weights, x_center, y_center):
    terms = np.zeros((3, 3))
    sum_w_rvd = 0.0
    sum_w_rvd_delA = 0.0
    sum_w_rvd_delB = 0.0

    for i in range(velocity_field.shape[1]):
        for j in range(velocity_field.shape[0]):
            vd = velocity_field[j, i]
            xp = x_field[j, i] - x_center
            yp = y_field[j, i] - y_center
            r = np.hypot(x_field[j, i], y_field[j, i])
            w = weights[j, i]
            
            delA = xp * xp + x_center * xp + yp * yp + y_center * yp
            delB = y_center * xp - x_center * yp

            terms[0, 0] += w
            terms[0, 1] += w * delA
            terms[0, 2] += w * delB
            terms[1, 1] += w * delA * delA
            terms[1, 2] += w * delA * delB
            terms[2, 2] += w * delB * delB
            
            # Y vector (LLSD eq 8)
            w_rvd = r * vd * w
            sum_w_rvd += w_rvd
            sum_w_rvd_delA += w_rvd * delA
            sum_w_rvd_delB += w_rvd * delB
            
    # Matching terms
    terms[1, 0] = terms[0, 1]
    terms[2, 0] = terms[0, 2]
    terms[2, 1] = terms[1, 2]

    # Matrix solve
    Y = np.array([sum_w_rvd, sum_w_rvd_delA, sum_w_rvd_delB])
    X = np.linalg.solve(terms, Y)

    return X[2]

@numba.njit()
def vorticity_llsd_dvad(vel_field, x_2d, y_2d, neighborhood_radius=1500, max_rays=51, range_delta=250):
    out = np.full_like(vel_field, np.nan)
    for nray in range(1, out.shape[0] - 1):
        for ngate in range(1, out.shape[1] - 1):
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
                if nray + j >= out.shape[0]:
                    within_j_bounds = False
                    break
                neighborhood_distance = np.hypot(x_2d[nray + j] - x_center, y_2d[nray + j] - y_center)
                if np.all(neighborhood_distance > neighborhood_radius):
                    # this is outer bound...if not, continue until next
                    break
            if not within_j_bounds:
                continue
            j_min = nray - j + 1
            j_max = nray + j
            
            # secondary bound check
            if i_min < 0 or j_min < 0 or i_max > out.shape[1] or j_max > out.shape[0]:
                continue
                
            # Calculate weights from circle
            neighborhood_distance = np.hypot(x_2d[j_min:j_max, i_min:i_max] - x_center, y_2d[j_min:j_max, i_min:i_max] - y_center)
            weights = (neighborhood_distance <= neighborhood_radius) * 1

            # Calculate value
            out[nray, ngate] = dvad_vorticity_local(
                vel_field[j_min:j_max, i_min:i_max],
                x_2d[j_min:j_max, i_min:i_max],
                y_2d[j_min:j_max, i_min:i_max],
                weights,
                x_center,
                y_center
            )
    
    return out
