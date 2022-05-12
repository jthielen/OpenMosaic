# Copyright (c) 2022 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Echo Top Height underlying numerical routines."""

import numba
import numpy as np


def echo_top_height(reflectivity, height, threshold, axis):
    # TODO: docstring
    out_shape = list(reflectivity.shape)
    del out_shape[axis]
    out = np.empty(out_shape, dtype=reflectivity.dtype)
    _echo_top_height_numba(reflectivity, height, out, threshold=threshold, axis=axis)
    return out


@numba.guvectorize('(n),(n)->()', nopython=True)
def _echo_top_height_numba(reflectivity, height, out, *, threshold=18):
    # TODO docstring
    out = np.nan
    if reflectivity[-1] >= threshold:
        # If our top of grid is above threshold, that's the height
        out = height[-1]
    else:
        # Otherwise, step down through column
        for i in range(len(reflectivity) - 1, 0, -1):
            if reflectivity[i - 1] >= threshold:
                # Once we hit threshold, get interpolated height
                out = (
                    height[i] * (reflectivity[i - 1] - threshold)
                    + height[i - 1] * (threshold - reflectivity[i])
                ) / (reflectivity[i - 1] - reflectivity[i])
