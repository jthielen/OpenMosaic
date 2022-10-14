# Copyright (c) 2022 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Underlying numerical routines for GridRad-style finite difference derivative moments.

TODO: implement
"""

import numpy as np
from scipy.signal import medfilt2d


def gridrad_filter(velocity):
    # 3x3 median filter
    velocity = medfilt2d(velocity, 3)
    # boxcar rolling mean in range (size 5)
    new_velocity = np.full_like(velocity, np.nan)
    new_velocity[:, 2:-2] = np.mean(np.stack([
        velocity[..., slice(i + 2, velocity.shape[1] + i - 2)] for i in range(-2, 3)
    ]), axis=0)
    return new_velocity

def azi_shear_smoothed_diff(velocity, range_2d, azi_2d):
    azishear = np.full_like(velocity, np.nan)
    azishear[1:-1] = (velocity[:-2] - velocity[2:]) / ((azi_2d[2:] - azi_2d[:-2]) * range_2d[1:-1])
    return azishear

def rad_div_smoothed_diff(velocity, range_2d, azi_2d):
    azishear = np.full_like(velocity, np.nan)
    azishear[:, 1:-1] = (velocity[:, :-2] - velocity[:, 2:]) / (range_2d[:, :-2] - range_2d[:, 2:])
    return azishear
