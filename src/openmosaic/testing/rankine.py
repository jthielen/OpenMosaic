# Copyright (c) 2022 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Rankine Vortex (and 'Burst', as its dual) simulation.

TODO: refactor for better api and to build in noisy field
      also add docstrings
"""

import metpy.calc as mpcalc
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
    vortex_intensity,
    min_x,
    max_x,
    min_y,
    max_y,
    delta_range=2.5e2,
    delta_azimuth=0.5,
    delta_cartesian=2.5e2,
    invert=False,
    beam_on_axis=True,
    velocity_stdev=1.5
):
    """
    Simulate a Rankine vortex (or analogous divergence signature) centered in domain.
    
    Parameters
    ----------
    vortex_radius : float
        Control radius for signature size
    vortex_intensity : float
        Maximum speed in output vortex signature. Can be set negative for anticyclonic or
        convergence signature.
    min_x, max_x, min_y, max_y : float
        Bounds of rectangular data, which also specify range to fill with polar data
    delta_range : float, optional
        Grid spacing in polar coordinate range direction
    delta_azimuth : float, optional
        Grid spacing in polar coordinate azimuth direction [in units of deg]
    delta_cartesian : float, optional
        Grid spacing for reference cartesian grid
    invert : bool, optional
        If true, use the usual tangential speed of Rankine vortex as radial instead to
        create a divergence signature.
    beam_on_axis : bool, optional
        Ensure that a beam is centered on central point of domain. Otherwise, have point on
        boundary between two beams.
    velocity_stdev : float, optional
        Standard deviation for Gaussian noise added to the simulated Doppler velocity.
    """
    r_corners = np.array([
        np.hypot(y, x) for y, x in (
            (min_y, min_x), (min_y, max_x), (max_y, min_x), (max_y, max_x)
        )
    ])
    theta_corners = np.array([
        np.arctan2(y, x) for y, x in (
            (min_y, min_x), (min_y, max_x), (max_y, min_x), (max_y, max_x)
        )
    ])
    if min_y * max_y <= 0 and min_x * min_x <= 0:
        # contains the origin
        raise ValueError("Cannot create grid that contains origin")
    elif min_y * max_y > 0 and min_x * min_y > 0:
        # does not straddle an axis
        min_range = r_corners.min()
        max_range = r_corners.max()
        min_azimuth = theta_corners.min()
        max_azimuth = theta_corners.max()
    elif max_x < 0 and max_y > 0:
        # straddles negative x-axis (choose different removable discontinuity)
        min_range = -max_x
        max_range = r_corners.max()
        min_azimuth = np.mod(theta_corners, 2 * np.pi).min()
        max_azimuth = np.mod(theta_corners, 2 * np.pi).max()
    else:
        if min_x > 0 and max_y > 0:
            # positive x-axis
            min_range = min_x
        elif max_y < 0:
            # negative y-axis
            min_range = -max_y
        else:
            min_range = min_y
        max_range = r_corners.max()
        min_azimuth = theta_corners.min()
        max_azimuth = theta_corners.max()
        
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
    vortex_center_x = (max_x + min_x) / 2
    vortex_center_y = (max_y + min_y) / 2
    vortex_distance_rtheta = np.hypot(x_rtheta - vortex_center_x, y_rtheta - vortex_center_y)
    vortex_distance_xy = np.hypot(x_2d - vortex_center_x, y_2d - vortex_center_y)
    speed_rtheta = rankine_vortex_speed_unadjusted(vortex_distance_rtheta, vortex_radius)
    speed_xy = rankine_vortex_speed_unadjusted(vortex_distance_xy, vortex_radius)
    speed_rtheta = vortex_intensity * speed_rtheta / speed_xy.max()  # normalize with common
                                                                     # max
    speed_xy = vortex_intensity * speed_xy / speed_xy.max()  # normalize with common max
    if invert:
        vector_direction_rtheta = np.arctan2(
            y_rtheta - vortex_center_y, x_rtheta - vortex_center_x
        )
        vector_direction_xy = np.arctan2(y_2d - vortex_center_y, x_2d - vortex_center_x)
    else:
        vector_direction_rtheta = np.arctan2(
            x_rtheta - vortex_center_x, -y_rtheta + vortex_center_y
        )
        vector_direction_xy = np.arctan2(x_2d - vortex_center_x, -y_2d + vortex_center_y)       
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
    stretching_deformation = mpcalc.stretching_deformation(
        u_xy * units('m/s'),
        v_xy * units('m/s'),
        dx=delta_cartesian * units.m,
        dy=delta_cartesian * units.m
    ).m
    shearing_deformation = mpcalc.shearing_deformation(
        u_xy * units('m/s'),
        v_xy * units('m/s'),
        dx=delta_cartesian * units.m,
        dy=delta_cartesian * units.m
    ).m
    
    # Compute radial velocity
    radial_velocity = u_rtheta * np.cos(theta_2d) + v_rtheta * np.sin(theta_2d)
    if velocity_stdev > 0:
        radial_velocity += np.random.normal(0, velocity_stdev, radial_velocity.shape)
    
    # Assemble output
    return xr.Dataset(
        {
            'radial_velocity': (('azimuth', 'range'), radial_velocity, {'units': 'm/s'}),
            'u': (('y', 'x'), u_xy, {'units': 'm/s'}),
            'v': (('y', 'x'), v_xy, {'units': 'm/s'}),
            'divergence': (('y', 'x'), divergence, {'units': '1/s'}),
            'vorticity': (('y', 'x'), vorticity, {'units': '1/s'}),
            'stretching_deformation': (('y', 'x'), stretching_deformation, {'units': '1/s'}),
            'shearing_deformation': (('y', 'x'), shearing_deformation, {'units': '1/s'}),
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
