# Copyright (c) 2022 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""The central GridDefinition class and supporting utils."""

import warnings

import numpy as np
import xarray as xr


attrs_to_use = {'units', 'standard_name', 'long_name'}


class GridDefinition:
    """TODO: basically everything...also may need to reconsider how subgrids are handled

    NEED TO ADD:
        - simplified (class method to just set bare minimum of grid specifications, and let
          rest be auto-determined by what radar data is input)
    
    
    """

    def __init__(self) -> None:
        # TODO
        # self.cf_projection_attrs
        pass

    def from_xarray(self, dataset_like):
        # TODO
        pass

    def get_coordinate(self, coordinate_name):
        # TODO
        # options: 'z', 'y', 'x', 'latitude', 'longitude'
        pass

    def combine_subgrids(
        self,
        *subgrids,
        include_weights_in_output=True,
        include_obs_count=False,
        include_longitude_latitude=False
    ):
        """TODO doc. Returns xarray Dataset"""
        # Use first subgrid to get field count and other config (TODO, need to validate?)
        prototype = subgrids[0]
        grid_shape = (
            self.grid_params['nz'],
            self.grid_params['ny'],
            self.grid_params['nx'],
            len(prototype.fields)
        )

        # Set up collection arrays
        grid_sum = np.zeros(grid_shape, dtype='float32')
        grid_wsum = np.zeros(grid_shape, dtype='float32')
        if prototype.include_echo_count:
            grid_n_echo = np.zeros(grid_shape[:-1], dtype='uint8')
        if include_obs_count:
            grid_n_obs = np.zeros(grid_shape[:-1], dtype='uint8')

        # Loop over all subgrids
        for subgrid in subgrids:
            grid_sum[:, subgrid.yi_slice, subgrid.xi_slice, :] += subgrid.sum
            grid_wsum[:, subgrid.yi_slice, subgrid.xi_slice, :] += subgrid.wsum
            if prototype.include_echo_count:
                grid_n_echo[:, subgrid.yi_slice, subgrid.xi_slice] += subgrid.n_echo
            if include_obs_count:
                grid_n_obs[:, subgrid.yi_slice, subgrid.xi_slice] += 1

        # Apply the weighting and masking
        mweight = np.ma.masked_equal(grid_wsum, 0)
        msum = np.ma.masked_array(grid_sum, mweight.mask)

        # Build xarray.Variable for each field
        data_vars = {}
        for i, f in enumerate(prototype.fields):
            data_vars[f] = xr.Variable(
                ('z', 'y', 'x'),
                msum[..., i] / mweight[..., i],
                {k: v for k, v in prototype.radar.fields[f].items() if k in attrs_to_use}
            )
        if include_weights_in_output:
            data_vars['mosaic_weights'] = xr.Variable(
                ('z', 'y', 'x'),
                mweight[..., 0],
                {
                    'description': (
                        'Sum of gridding weights from chosen gridding procedure (see dataset '
                        'description for details).'
                    )
                }
            )
        if prototype.include_echo_count:
            data_vars['mosaic_n_echo'] = xr.Variable(
                ('z', 'y', 'x'),
                grid_n_echo,
                {
                    'description': (
                        'Number of valid radar echoes contributing to this grid volume'
                    )
                }
            )
        if include_obs_count:
            data_vars['mosaic_n_obs'] = xr.Variable(
                ('z', 'y', 'x'),
                grid_n_obs,
                {
                    'description': (
                        'Number of separate Level II volumes contributing to this grid volume'
                    )
                }
            )
        
        # Go back through all those data vars and add the grid_mapping
        for varname in data_vars:
            data_vars[varname].attrs['grid_mapping'] = 'projection'
        data_vars['projection'] = xr.Variable(tuple(), 0, self.cf_projection_attrs)

        # Construct coords
        coord_vars = {
            'time': xr.Variable(
                tuple(), prototype.analysis_time, {'long_name': 'Time of radar analysis'}
            ),
            'z': self.get_coordinate('z').variable,
            'y': self.get_coordinate('y').variable,
            'x': self.get_coordinate('x').variable
        }
        if include_longitude_latitude:
            coord_vars['longitude'] = self.get_coordinate('longitude').variable
            coord_vars['latitude'] = self.get_coordinate('latitude').variable

        return xr.Dataset(data_vars, coord_vars)

        




class RadarSubgrid:
    """TODO"""

    def __init__(self, radar, parent_grid):
        # TODO
        # self.fields
        # self.radar
        # self.include_echo_count = ...
        # self.sum
        # self.wsum
        # self.yi_slice
        # self.xi_slice
        # self.n_echo
        # self.analysis_time
        pass

    