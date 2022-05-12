# Copyright (c) 2022 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Underlying numerical routines for MRMS-style linear-least squares derivative moments.

TODO: refactor to share with DVAD-LLSD approach(es) as well.
      also docstrings
"""

import numba
import numpy as np


# LLSD
@numba.njit()
def azi_shear_local_linalg(radar_field, weights, m, n, r, r_delta, azi_delta):
    # init accumulating terms (M and Y from LLSD eq 8 corrected, adj(M) eq 10)
    terms = np.zeros((3, 3))
    sum_w_r_u = 0.0
    sum_w_theta_u = 0.0
    sum_w_u = 0.0

    for i in range(m):
        r_dist = r_delta * (i - (m // 2))
        for j in range(n):
            u = radar_field[j, i]
            w = weights[j, i]
            theta_dist = azi_delta * (j - (n // 2)) * (r + r_dist)

            # M matrix (LLSD eq 8 corrected)
            terms[0, 0] += w * r_dist * theta_dist
            terms[0, 1] += w * r_dist * r_dist
            terms[0, 2] += w * r_dist
            terms[1, 0] += w * theta_dist * theta_dist
            terms[1, 2] += w * theta_dist
            terms[2, 2] += w
            
            # Y vector (LLSD eq 8)
            sum_w_r_u += w * r_dist * u
            sum_w_theta_u += w * theta_dist * u
            sum_w_u += w * u
            
    # Matching terms
    terms[1, 1] = terms[0, 0]
    terms[2, 1] = terms[0, 2]
    terms[2, 0] = terms[1, 2]

    # Matrix solve
    Y = np.array([sum_w_r_u, sum_w_theta_u, sum_w_u])
    X = np.linalg.solve(terms, Y)

    return X[0]


@numba.njit()
def azi_shear_llsd(vel_field, range_2d, azi_2d, azi_width=2500, r_depth=750, azi_max_rays=51):
    out = np.full_like(vel_field, np.nan)
    for nray in range(1, out.shape[0] - 1):
        azi_delta_1d = np.mod(azi_2d[nray + 1] - azi_2d[nray - 1], 2 * np.pi) / 2    
        r = range_2d[nray]
        for ngate in range(1, out.shape[1] - 1):
            if 2.0 * (r[ngate] - r[0]) < r_depth or 2.0 * (r[-1] - r[ngate]) < r_depth:
                # Boundary, computation invalid
                continue
            r_delta = (r[ngate + 1] - r[ngate - 1]) / 2.0
            azi_delta = azi_delta_1d[ngate]

            # Compute kernel size
            half_m_float = r_depth / (2.0 * r_delta)
            m = 2 * int(half_m_float) + 1
            if m < 3:
                m = 3
            half_n_float = azi_width / (2.0 * azi_delta * r[ngate])
            n = 2 * int(half_n_float) + 1
            if n < 3:
                n = 3
            elif n > azi_max_rays:
                if azi_max_rays % 2 == 0:
                    n = azi_max_rays - 1
                else:
                    n = azi_max_rays

            # Compute kernel bounds into array
            i_min = ngate - (m // 2)
            i_max = ngate + (m // 2) + 1
            j_min = nray - (n // 2)
            j_max = nray + (n // 2) + 1
            
            # secondary bound check
            if i_min < 0 or j_min < 0 or i_max > out.shape[1] or j_max > out.shape[0]:
                continue

            # Calculate value
            out[nray, ngate] = azi_shear_local_linalg(
                vel_field[j_min:j_max, i_min:i_max],
                np.ones((n, m)),
                m,
                n,
                r[ngate],
                r_delta,
                -azi_delta
            )
    
    return out