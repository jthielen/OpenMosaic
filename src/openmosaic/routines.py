# Copyright (c) 2020 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Implementations of field calculations."""

import numpy as np

from ._c_routines import azi_shear_uniform_weight, div_shear_uniform_weight


def compute_azimuthal_shear(
    radar,
    velocity_key,
    azi_width=2500.,
    r_depth=750.,
    azi_max_rays=51
):
    """Compute azimuthal shear over each sweep of the given radar, appending the field.

    Uses the formulation given by Mahalik et al. (2019), Equation 12a, with uniform (unity)
    weighting.

    Parameters
    ----------
    radar : pyart.core.Radar
        Py-ART radar object
    velocity_key : str
        Field name for filtered/dealiased velocity field
    azi_width : float, optional
        Kernel width in azimuthal direction, in meters. Defaults to 2500 m.
    r_depth : float, optional
        Kernel depth in range direction, in meters. Defaults to 750 m.
    azi_max_rays : int, optional
        Maximum number of rays to include in kernel (so as to not subtend too large an angle
        when near the radar site). Defaults to 51.
    """

    azi_shear = np.zeros((radar.nrays, radar.ngates), dtype=np.dtype('float32'))
    r = radar.range['data']

    for nsweep in range(radar.nsweeps):
        vel_field = radar.get_field(nsweep, velocity_key, False)
        theta = np.deg2rad(radar.get_azimuth(nsweep, False))
        azi_shear[radar.get_slice(nsweep), :] = azi_shear_uniform_weight(
            r.shape[0],
            theta.shape[0],
            r.astype('float32'),
            theta.astype('float32'),
            vel_field.astype('float32'),
            azi_width,
            r_depth,
            azi_max_rays
        )

    field = {
        'data': azi_shear,
        'units': '1 / second',
        'long_name': 'azimuthal_shear',
        'coordinates': 'elevation azimuth range'
    }
    radar.add_field('azi_shear', field, True)
    return field


def compute_radial_divergence(
    radar,
    velocity_key,
    azi_width=750.,
    r_depth=1500.,
    azi_max_rays=17
):
    """Compute radial divergence over each sweep of the given radar, appending the field.

    Uses the formulation given by Mahalik et al. (2019), Equation 12b, with uniform (unity)
    weighting.

    Parameters
    ----------
    radar : pyart.core.Radar
        Py-ART radar object
    velocity_key : str
        Field name for filtered/dealiased velocity field
    azi_width : float, optional
        Kernel width in azimuthal direction, in meters. Defaults to 750 m.
    r_depth : float, optional
        Kernel depth in range direction, in meters. Defaults to 1500 m.
    azi_max_rays : int, optional
        Maximum number of rays to include in kernel (so as to not subtend too large an angle
        when near the radar site). Defaults to 17.
    """

    div_shear = np.zeros((radar.nrays, radar.ngates), dtype=np.dtype('float32'))
    r = radar.range['data']

    for nsweep in range(radar.nsweeps):
        vel_field = radar.get_field(nsweep, velocity_key, False)
        theta = np.deg2rad(radar.get_azimuth(nsweep, False))
        div_shear[radar.get_slice(nsweep), :] = div_shear_uniform_weight(
            r.shape[0],
            theta.shape[0],
            r,
            theta,
            vel_field,
            azi_width,
            r_depth,
            azi_max_rays
        )

    field = {
        'data': div_shear,
        'units': '1 / second',
        'long_name': 'radial_divergence',
        'coordinates': 'elevation azimuth range'
    }
    radar.add_field('rad_divergence', field, True)
    return field
