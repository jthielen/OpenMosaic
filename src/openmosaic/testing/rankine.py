# Copyright (c) 2022 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Rankine Vortex (and 'Burst', as its dual) simulation.

TODO: refactor for better api and to build in noisy field
      also add docstrings
"""

import numba
from numba import float64
import numpy as np
import xarray as xr


@numba.vectorize([float64(float64, float64)], nopython=True)
def rankine_vortex_speed_unadjusted(radius, control_radius):
    if radius > control_radius:
        return 1 / (2 * np.pi * radius)
    else:
        return radius / (2 * np.pi * control_radius**2)


def rankine_simulate(
    vortex_radius,
    vortex_displacement,
    vortex_intensity,
    min_x,
    max_x,
    min_y,
    max_y,
    delta_range=2.5e2,
    delta_azimuth=0.5,
    delta_cartesian=2.5e2,
    invert=False,
    beam_on_axis=True
):
    """
    Simulate a Rankine vortex (or analogous divergence signature) along x-axis.
    
    Parameters
    ----------
    vortex_radius : float
        Control radius for signature size
    vortex_displacement : float
        Distance of centroid of signature along x-axis
    vortex_intensity : float
        Maximum speed in output vortex signature. Can be set negative for anticyclonic or
        convergence signature.
    min_x, max_x, min_y, max_y : float
        Bounds of rectangular data, which also specify range to fill with polar data
    delta_range : float
        Grid spacing in polar coordinate range direction
    delta_azimuth : float
        Grid spacing in polar coordinate azimuth direction [in units of deg]
    delta_cartesian : float
        Grid spacing for reference cartesian grid
    invert : bool
        If true, use the usual tangential speed of Rankine vortex as radial instead to
        create a divergence signature.
    beam_on_axis : bool
        Ensure that a beam is centered on x-axis. Otherwise, have x-axis on boundary
        between two beams.
    """
    # optional import
    try:
        import metpy.calc as mpcalc
        from metpy.units import units
    except ImportError:
        raise ValueError("Rankine simulator requires MetPy.")

    # Solve for bounds for grid creation
    if min_y * max_y > 0:
        raise ValueError("Cannot create grid that does not encompass the x-axis")
    if min_x >= 0:
        min_azimuth = np.arctan2(min_y, min_x)
        max_azimuth = np.arctan2(max_y, min_x)
        min_range = min_x
        max_range = max(np.hypot(max_y, max_x), np.hypot(min_y, max_x))
    elif max_x <= 0:
        min_azimuth = np.arctan2(max_y, max_x)
        max_azimuth = 2 * np.pi + np.arctan(min_y, max_x)
        min_range = -max_x
        max_range = max(np.hypot(max_y, min_x), np.hypot(min_y, min_x))
    else:
        min_azimuth = -np.pi
        max_azimuth = np.pi
        min_range = 0
        max_range = max(
            np.hypot(y, x) for y, x in (
                (min_y, min_x), (min_y, max_x), (max_y, min_x), (max_y, max_x)
            )
        )
        
    # Actually calculate the grid coords
    x = np.arange(min_x, max_x + delta_cartesian, delta_cartesian)
    y = np.arange(min_y, max_y + delta_cartesian, delta_cartesian)
    r = np.arange(min_range, max_range + delta_range, delta_range)
    delta_azimuth = np.deg2rad(delta_azimuth)
    theta = np.concatenate([
        np.arange(
            0 if beam_on_axis else -delta_azimuth/2,
            min_azimuth - delta_azimuth * 5,
            -delta_azimuth
        )[::-1],
        np.arange(
            delta_azimuth if beam_on_axis else delta_azimuth/2,
            max_azimuth + delta_azimuth * 5,
            delta_azimuth
        )
    ])
    x_2d, y_2d = np.meshgrid(x, y)
    r_2d, theta_2d = np.meshgrid(r, theta)
    x_rtheta = r_2d * np.cos(theta_2d)
    y_rtheta = r_2d * np.sin(theta_2d)
    
    # Compute the velocities on both grids
    vortex_distance_rtheta = np.hypot(x_rtheta - vortex_displacement, y_rtheta)
    vortex_distance_xy = np.hypot(x_2d - vortex_displacement, y_2d)
    speed_rtheta = rankine_vortex_speed_unadjusted(vortex_distance_rtheta, vortex_radius)
    speed_xy = rankine_vortex_speed_unadjusted(vortex_distance_xy, vortex_radius)
    speed_rtheta = vortex_intensity * speed_rtheta / speed_xy.max()  # normalize with common max
    speed_xy = vortex_intensity * speed_xy / speed_xy.max()  # normalize with common max
    if invert:
        vector_direction_rtheta = np.arctan2(y_rtheta, x_rtheta - vortex_displacement)
        vector_direction_xy = np.arctan2(y_2d, x_2d - vortex_displacement)
    else:
        vector_direction_rtheta = np.arctan2(x_rtheta - vortex_displacement, -y_rtheta)
        vector_direction_xy = np.arctan2(x_2d - vortex_displacement, -y_2d)       
    u_rtheta = speed_rtheta * np.cos(vector_direction_rtheta)
    v_rtheta = speed_rtheta * np.sin(vector_direction_rtheta)
    u_xy = speed_xy * np.cos(vector_direction_xy)
    v_xy = speed_xy * np.sin(vector_direction_xy)
    
    # Compute divergence and vorticity
    divergence = mpcalc.divergence(
        u_xy * units('m/s'),
        v_xy * units('m/s'),
        dx=delta_cartesian * units.m,
        dy=delta_cartesian * units.m
    ).m
    vorticity = mpcalc.vorticity(
        u_xy * units('m/s'),
        v_xy * units('m/s'),
        dx=delta_cartesian * units.m,
        dy=delta_cartesian * units.m
    ).m
    
    # Compute radial velocity
    radial_velocity = u_rtheta * np.cos(theta_2d) + v_rtheta * np.sin(theta_2d)
    
    # Assemble output
    return xr.Dataset(
        {
            'radial_velocity': (('azimuth', 'range'), radial_velocity, {'units': 'm/s'}),
            'u': (('y', 'x'), u_xy, {'units': 'm/s'}),
            'v': (('y', 'x'), v_xy, {'units': 'm/s'}),
            'divergence': (('y', 'x'), divergence, {'units': '1/s'}),
            'vorticity': (('y', 'x'), vorticity, {'units': '1/s'})
        },
        {
            'azimuth': (('azimuth',), np.rad2deg(theta), {'units': 'degrees'}),
            'range': r,
            'y': y,
            'x': x,
            'x_polar': (('azimuth', 'range'), x_rtheta),
            'y_polar': (('azimuth', 'range'), y_rtheta)
        }
    )
