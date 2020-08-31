"""
Python utilities vendored from Py-ART to adapt underlying interpolation routine for
GridRad procedures.

This file and the associated _gate_to_grid_map.pyx file are copied with modification
from the Py-ART package according to the terms of its license (which is reproduced
below). All other modifications are Copyright (c) 2020 by the OpenMosaic developers,
and released under the terms of the Apache License, Version 2 (see LICENSE in main
project directory).

-----------------------------------------------------------------------------------------

Copyright (c) 2013, UChicago Argonne, LLC
All rights reserved.

Copyright 2013 UChicago Argonne, LLC. This software was produced under U.S. 
Government contract DE-AC02-06CH11357 for Argonne National Laboratory (ANL),
which is operated by UChicago Argonne, LLC for the U.S. Department of Energy.
The U.S. Government has rights to use, reproduce, and distribute this
software.  NEITHER THE GOVERNMENT NOR UCHICAGO ARGONNE, LLC MAKES ANY 
WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS
SOFTWARE.  If software is modified to produce derivative works, such modified
software should be clearly marked, so as not to confuse it with the version
available from ANL.
 
Additionally, redistribution and use in source and binary forms, with or 
without modification, are permitted provided that the following conditions
are met:
    
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of UChicago Argonne, LLC, Argonne National
      Laboratory, ANL, the U.S. Government, nor the names of its 
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY UCHICAGO ARGONNE, LLC AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL UCHICAGO ARGONNE, LLC OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

# Import statements modified to use Py-ART as third party
import warnings

from cftime import num2date  # OpenMosaic addition
import numpy as np

from pyart.core.radar import Radar
from pyart.core.transforms import geographic_to_cartesian
from pyart.filters import GateFilter, moment_based_gate_filter

from ._gate_to_grid_map import GateToGridMapper
from ._map_gates_to_subgrid import RoIFunction, ConstantRoI, DistBeamRoI, DistRoI


def map_gates_to_grid(
        radars, grid_shape, grid_limits, grid_origin=None,
        grid_origin_alt=None, grid_projection=None,
        fields=None, gatefilters=False, map_roi=True,
        weighting_function='Barnes', toa=17000.0, roi_func='dist_beam',
        constant_roi=None, z_factor=0.05, xy_factor=0.02, min_radius=500.0,
        h_factor=1.0, nb=1.5, bsp=1.0, analysis_time=None, **kwargs):
    """
    Map gates from one or more radars to a Cartesian grid.

    Generate a Cartesian grid of points for the requested fields from the
    collected points from one or more radars. For each radar gate that is not
    filtered a radius of influence is calculated. The weighted field values
    for that gate are added to all grid points within that radius. This
    routine scaled linearly with the number of radar gates and the effective
    grid size.

    Parameters not defined below are identical to those in
    :py:func:`map_to_grid`.

    Parameters
    ----------
    roi_func : str or RoIFunction
        Radius of influence function. A function which takes an
        z, y, x grid location, in meters, and returns a radius (in meters)
        within which all collected points will be included in the weighting
        for that grid points. Examples can be found in the
        Typically following strings can use to specify a built in
        radius of influence function:

            * constant: constant radius of influence.
            * dist: radius grows with the distance from each radar.
            * dist_beam: radius grows with the distance from each radar
              and parameter are based of virtual beam sizes.

        A custom RoIFunction can be defined using the RoIFunction class
        and defining a get_roi method which returns the radius. For efficient
        mapping this class should be implemented in Cython.

    Returns
    -------
    grids : dict
        Dictionary of mapped fields. The keys of the dictionary are given by
        parameter fields. Each elements is a `grid_size` float64 array
        containing the interpolated grid for that field.

    """
    # make a tuple if passed a radar object as the first argument
    if isinstance(radars, Radar):
        radars = (radars, )

    skip_transform = False
    if len(radars) == 1 and grid_origin_alt is None and grid_origin is None:
        skip_transform = True

    if grid_origin_alt is None:
        try:
            grid_origin_alt = float(radars[0].altitude['data'])
        except TypeError:
            grid_origin_alt = np.mean(radars[0].altitude['data'])

    gatefilters = _parse_gatefilters(gatefilters, radars)
    cy_weighting_function = _determine_cy_weighting_func(weighting_function)
    projparams = _find_projparams(grid_origin, radars, grid_projection)
    fields = _determine_fields(fields, radars)
    grid_starts, grid_steps = _find_grid_params(grid_shape, grid_limits)
    offsets = _find_offsets(radars, projparams, grid_origin_alt)
    roi_func = _parse_roi_func(roi_func, constant_roi, z_factor, xy_factor,
                               min_radius, h_factor, nb, bsp, offsets)

    # prepare grid storage arrays
    nfields = len(fields)
    grid_sum = np.zeros(grid_shape + (nfields, ), dtype=np.float32)
    grid_wsum = np.zeros(grid_shape + (nfields, ), dtype=np.float32)
    gatemapper = GateToGridMapper(
        grid_shape, grid_starts, grid_steps, grid_sum, grid_wsum)

    # project gates from each radar onto the grid
    for radar, gatefilter in zip(radars, gatefilters):

        # Copy the field data and masks.
        # TODO method that does not copy field data into new array
        shape = (radar.nrays, radar.ngates, nfields)
        field_data = np.empty(shape, dtype='float32')
        field_mask = np.empty(shape, dtype='uint8')
        for i, field in enumerate(fields):
            fdata = radar.fields[field]['data']
            field_data[:, :, i] = np.ma.getdata(fdata)
            field_mask[:, :, i] = np.ma.getmaskarray(fdata)

        # find excluded gates from the gatefilter
        if gatefilter is False:
            gatefilter = GateFilter(radar)  # include all gates
        elif gatefilter is None:
            gatefilter = moment_based_gate_filter(radar, **kwargs)
        excluded_gates = gatefilter.gate_excluded.astype('uint8')

        # calculate gate locations relative to the grid origin
        if skip_transform:
            # single radar, grid centered at radar location
            gate_x = radar.gate_x['data']
            gate_y = radar.gate_y['data']
        else:
            gate_x, gate_y = geographic_to_cartesian(
                radar.gate_longitude['data'], radar.gate_latitude['data'],
                projparams)
        gate_z = radar.gate_altitude['data'] - grid_origin_alt

        # Range and offset params (OpenMosiac addition)
        gate_range = radar.range['data']
        if analysis_time is None:
            gate_timedelta = np.zeros(radar.nrays)
        else:
            gate_timedelta = np.array([
                (t - analysis_time.replace(tzinfo=None)).total_seconds()
                for t in num2date(radar.time['data'], radar.time['units'])
            ])

        # map the gates onto the grid (modified with OpenMosaic addition)
        gatemapper.map_gates_to_grid(
            radar.ngates, radar.nrays, gate_z.astype('float32'),
            gate_y.astype('float32'), gate_x.astype('float32'),
            gate_range.astype('float32'), gate_timedelta.astype('float32'),
            field_data, field_mask, excluded_gates,
            toa, roi_func, cy_weighting_function)

    # create and return the grid dictionary
    mweight = np.ma.masked_equal(grid_wsum, 0)
    msum = np.ma.masked_array(grid_sum, mweight.mask)
    grids = dict(
        [(f, msum[..., i] / mweight[..., i]) for i, f in enumerate(fields)])
    if map_roi:
        roi_array = np.empty(grid_shape, dtype=np.float32)
        gatemapper.find_roi_for_grid(roi_array, roi_func)
        grids['ROI'] = roi_array
    return grids


def _determine_cy_weighting_func(weighting_function):
    """ Determine cython weight function value. """
    if weighting_function.upper() == 'GRIDRAD':
        cy_weighting_function = 4
    elif weighting_function.upper() == 'BARNES2':
        cy_weighting_function = 3
    elif weighting_function.upper() == 'NEAREST':
        cy_weighting_function = 2
    elif weighting_function.upper() == 'CRESSMAN':
        cy_weighting_function = 1
    elif weighting_function.upper() == 'BARNES':
        warnings.warn("Barnes weighting function is deprecated."
                      " Please use Barnes 2 to be consistent with"
                      " Pauley and Wu 1990.", DeprecationWarning)
        cy_weighting_function = 0
    else:
        raise ValueError('unknown weighting_function')
    return cy_weighting_function


def _find_projparams(grid_origin, radars, grid_projection):
    """ Determine the projection parameter. """

    # parse grid_origin
    if grid_origin is None:
        try:
            lat = float(radars[0].latitude['data'])
            lon = float(radars[0].longitude['data'])
        except TypeError:
            lat = np.mean(radars[0].latitude['data'])
            lon = np.mean(radars[0].longitude['data'])
        grid_origin = (lat, lon)

    grid_origin_lat, grid_origin_lon = grid_origin

    # parse grid_projection
    if grid_projection is None:
        grid_projection = {
            'proj': 'pyart_aeqd', '_include_lon_0_lat_0': True}
    projparams = grid_projection.copy()
    if projparams.pop('_include_lon_0_lat_0', False):
        projparams['lon_0'] = grid_origin_lon
        projparams['lat_0'] = grid_origin_lat
    return projparams


def _parse_gatefilters(gatefilters, radars):
    """ Parse the gatefilters parameter. """
    if isinstance(gatefilters, GateFilter):
        gatefilters = (gatefilters, )  # make tuple if single filter passed
    if gatefilters is False:
        gatefilters = (False, ) * len(radars)
    if gatefilters is None:
        gatefilters = (None, ) * len(radars)
    if len(gatefilters) != len(radars):
        raise ValueError('Length of gatefilters must match length of radars')
    return gatefilters


def _determine_fields(fields, radars):
    """ Determine which field should be mapped to the grid. """
    if fields is None:
        fields = set(radars[0].fields.keys())
        for radar in radars[1:]:
            fields = fields.intersection(radar.fields.keys())
        fields = list(fields)
    return fields


def _find_offsets(radars, projparams, grid_origin_alt):
    """ Find offset between radars and grid origin. """
    # loop over the radars finding offsets from the origin
    offsets = [] # offsets from the grid origin, in meters, for each radar
    for radar in radars:
        x_disp, y_disp = geographic_to_cartesian(
            radar.longitude['data'], radar.latitude['data'], projparams)
        try:
            z_disp = float(radar.altitude['data']) - grid_origin_alt
            offsets.append((z_disp, float(y_disp), float(x_disp)))
        except TypeError:
            z_disp = np.mean(radar.altitude['data']) - grid_origin_alt
            offsets.append((z_disp, np.mean(y_disp), np.mean(x_disp)))
    return offsets


def _find_grid_params(grid_shape, grid_limits):
    """ Find the starting points and step size of the grid. """
    nz, ny, nx = grid_shape
    zr, yr, xr = grid_limits
    z_start, z_stop = zr
    y_start, y_stop = yr
    x_start, x_stop = xr

    if nz == 1:
        z_step = 0.
    else:
        z_step = (z_stop - z_start) / (nz - 1.)
    if ny == 1:
        y_step = 0.
    else:
        y_step = (y_stop - y_start) / (ny - 1.)
    if nx == 1:
        x_step = 0.
    else:
        x_step = (x_stop - x_start) / (nx - 1.)

    grid_starts = (z_start, y_start, x_start)
    grid_steps = (z_step, y_step, x_step)
    return grid_starts, grid_steps


def _parse_roi_func(roi_func, constant_roi, z_factor, xy_factor, min_radius,
                    h_factor, nb, bsp, offsets):
    """ Return the Radius of influence object. """
    if not isinstance(roi_func, RoIFunction):
        if constant_roi is not None:
            roi_func = 'constant'
        else:
            constant_roi = 500.0
        if roi_func == 'constant':
            roi_func = ConstantRoI(constant_roi)
        elif roi_func == 'dist':
            roi_func = DistRoI(z_factor, xy_factor, min_radius, offsets)
        elif roi_func == 'dist_beam':
            roi_func = DistBeamRoI(h_factor, nb, bsp, min_radius, offsets)
        else:
            raise ValueError('unknown roi_func: %s' % roi_func)
    return roi_func
