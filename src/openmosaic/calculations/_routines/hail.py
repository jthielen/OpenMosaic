# Copyright (c) 2022 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""SHI and MESH underlying numerical routines."""

import numba
import numpy as np


def shi(reflectivity, height, melting_altitude, m20_altitude, axis):
    # TODO: docstring
    out_shape = list(melting_altitude.shape)
    storm_top_height = np.empty(out_shape, dtype=reflectivity.dtype)
    out = np.empty(out_shape, dtype=reflectivity.dtype)
    _echo_top_height_numba(reflectivity, height, storm_top_height, axis=axis)
    _shi_numba(
        reflectivity, height, storm_top_height, melting_altitude, m20_altitude, out, axis=axis
    )
    return out


def mesh(reflectivity, height, melting_altitude, m20_altitude, axis):
    return np.sqrt(_shi(reflectivity, height, melting_altitude, m20_altitude, axis))


@numba.guvectorize('(n),(n),(),(),()->()', nopython=True)
def _shi_numba(reflectivity, height, storm_top_height, melting_altitude, m20_altitude, out):
    # Calculate W_Z (which has branches)
    W_Z = (reflectivity - 40.0) / 10.0
    W_Z[reflectivity <= 40] = 0.0
    W_Z[reflectivity >= 50] = 1.0

    # Calculate E_dot
    E_dot = 5e-6 * np.power(10, 0.084 * reflectivity) * W_Z

    # Calculate W_T_H_T
    if storm_top_height <= melting_altitude:
        W_T_H_T = 0.0
    elif storm_top_height >= m20_altitude:
        W_T_H_T = 1.0
    else:
        W_T_H_T = (storm_top_height - melting_altitude) / (m20_altitude - melting_altitude)

    out = 0.1 * W_T_H_T * np.trapz(E_dot, x=height)
