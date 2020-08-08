"""
Gridder class (and directly supporting utils) for efficient and parallelized mapping of
Py-ART Radar objects to a common, regular grid. Supported by the underlying subgrid mapping
of the GatesToSubgridMapper.

Portions inspired by the map_gates_to_grid implementation of Py-ART (license reproduced in
_map_gates_to_subgrid.pyx, which more directly reuses Py-ART code).

"""

import warnings

from cftime import num2date
import dask
import numpy as np
from pyart.core.radar import Radar
from pyart.filters import GateFilter, moment_based_gate_filter
import pyproj

from ._map_gates_to_subgrid import GatesToSubgridMapper
from .grid_utils import generate_rectangular_grid
from .vendored import (
    _determine_cy_weighting_func,
    _parse_gatefilters,
    _determine_fields,
    _parse_roi_func
)


_grid_param_labels_z = ['nz', 'dz', 'z_min', 'z_max']
_grid_param_labels_y = ['ny', 'dy', 'y_min', 'y_max']
_grid_param_labels_z = ['nx', 'dx', 'x_min', 'x_max']


@dask.delayed
def radar_coords_to_grid_coords(
    gate_radar_x,
    gate_radar_y,
    site_id,
    transformer,
    wait_for_cache=False
):
    """Map radar x/y to grid x/y using transformer or a cached regression of it."""
    # TODO: the whole thing
    return False


@dask.delayed
def map_gates_to_subgrid(
    subgrid_shape,
    subgrid_starts,
    subgrid_steps,
    field_shape,
    field_data,
    field_mask,
    excluded_gates,
    gate_z,
    gate_y,
    gate_x,
    gate_range,
    gate_timedelta,
    toa,
    roi_func,
    cy_weighting_function
):
    """Delayed functional wrapper of GatesToSubgridMapper class."""
    # TODO: the whole thing
    return False


class Gridder:
    """Map gates from one or more radars to a common grid.

    TODO: document

    """

    cache_dir = "/tmp/"
    grid_params = {}

    # Control params
    r_max = 3.0e5

    def __init__(self, cf_projection, **grid_params):
        """Set up Gridder by defining grid.

        Parameters
        ----------
        cf_projection : dict-like
            Dictionary of projection parameters as defined by the CF Conventions
        **grid_params
            Additional keyword arguments for controlling the grid points in the specified
            projected space. Four options for each dimension (z, y, x) are available, be sure
            to only specify three.
            - ``nz``: number of grid points
            - ``dz``: grid spacing
            - ``z_min``: inclusive lower bound
            - ``z_max``: inclusive upper bound
            (likewise for y and x) 

        """
        self.cf_attrs = dict(cf_projection)
        self.crs = pyproj.CRS.from_cf(self.cf_attrs)

        self._assign_grid_params('x', {k: v for k, v in kwargs.items() if k in _grid_param_lables_x})
        self._assign_grid_params('y', {k: v for k, v in kwargs.items() if k in _grid_param_lables_y})
        self._assign_grid_params('z', {k: v for k, v in kwargs.items() if k in _grid_param_lables_z})
        
    def assign_from_subbatch_and_spacing(self, subbatch, spacing):
        """Set horizontal grid params from subbatch and spacing in units of grid projection.

        Parameters
        ----------
        subbatch : pandas.Series or dict-like
            Subscriptable with keys 'x_min', 'x_max', 'y_min', and 'y_max'
        spacing : float
            Regular horizontal grid spacing in projection space units

        """
        self._assign_grid_params(
            'x',
            {'x_min': subbatch['x_min'], 'x_max': subbatch['x_max'], 'dx': spacing}
        )
        self._assign_grid_params(
            'y',
            {'y_min': subbatch['y_min'], 'y_max': subbatch['y_max'], 'dy': spacing}
        )

    def _assign_grid_params(self, dim, params):
        """Assign grid parameters for dimension."""
        n = params.get(f"n{dim}", None)
        d = params.get(f"d{dim}", None)
        min = params.get(f"{dim}_min", None)
        max = params.get(f"{dim}_max", None)

        if len([param for param in [n, d, min, max] if param is None]) != 1:
            raise ValueError(
                f"Exactly three of four grid parameters must be specified for a "
                f"well-defined grid for dimension {dim}."
            )

        if n is None:
            n = int((max - min) / d) + 1
            if np.abs(max - min + d * (n - 1)) > 1e-6:
                raise ValueError(f"Grid min and max not evenly separated by d{dim}.")
        elif d is None:
            d = (max - min) / (n - 1)
        elif min is None:
            min = max - d * (n - 1)
        elif max is None:
            max = min + d * (n - 1)

        self.grid_params[f"n{dim}"] = n
        self.grid_params[f"d{dim}"] = d
        self.grid_params[f"{dim}_min"] = min
        self.grid_params[f"{dim}_max"] = max

    def round_grid_params(self, dims=None):
        """Regularize grid params so that grid is periodic with a 0 origin.

        Grid spacing (dz, dy, dx) is the unmodified parameter, with z_min rounded down and
        z_max rounded up in order to fully encompass the originally specified region. Count is
        then recomputed with the new bounds and fixed spacing.

        Parameters
        ----------
        dims : iterable or None, optional
            Collection of 'x', 'y', and 'z' dims to regularize. Defaults to None, which rounds each dim.

        """

        dims = ('z', 'y', 'x') if dims is None else dims
        for dim in dims:
            self.grid_params[f"{dim}_min"] = (
                np.floor(self.grid_params[f"{dim}_min"] / self.grid_params[f"d{dim}"])
                * self.grid_params[f"d{dim}"]
            )
            self.grid_params[f"{dim}_max"] = (
                np.ceil(self.grid_params[f"{dim}_max"] / self.grid_params[f"d{dim}"])
                * self.grid_params[f"d{dim}"]
            )
            self.grid_params[f"n{dim}"] = int(
                (self.grid_params[f"{dim}_max"] - self.grid_params[f"{dim}_min"])
                / self.grid_params[f"d{dim}"]
            ) + 1

    def prepare_radars(radars, cache_dir=None):
        """Determine gate coordinates and subgrids on which to map each radar.

        Parameters
        ----------
        radars : iterable of pyart.core.radar.Radar
            Collection of (already subsetted by sweep time and range) Py-ART Radar objects
        cache_dir : str
            Path to cache directory (used to save coord transform regression models for each
            radar site)

        Notes
        -----
        This should run quickly since the computation of coordinates is delayed using Dask.
        Remember to run ``compute_coords`` afterwards
        """
        if cache_dir is not None:
            self.cache_dir = cache_dir

        self.radars = []
        radar_site_ids = []
        for radar in radars:
            # Define projection objects
            radar_crs = pyproj.CRS({
                'proj': 'aeqd',
                'lat_0': radar.latitude['data'].item(),
                'lon_0': radar.longitude['data'].item()
            })
            transformer = pyproj.Transformer.from_crs(radar_crs, self.crs)

            # Determine subset of grid that this radar will map data on to
            x_radar, y_radar = transformer.transform(0, 0)
            # Inclusive index of destination grid point to the left of left-most influence of
            # radar data
            xi_min = max(
                0,
                int((x_radar - r_max - self.grid_params['x_min']) / self.grid_params['dx'])
            )
            # Exclusive index (upper) of destination grid point to the right of the right-most
            # influence of radar data
            xi_max = min(
                self.grid_params['nx'],
                int(
                    (x_radar + r_max - self.grid_params['x_min'])
                    / self.grid_params['dx']
                ) + 2
            )
            # Inclusive index of destination grid point below the bottom-most influence of
            # radar data
            yi_min = max(
                0,
                int((y_radar - r_max - self.grid_params['y_min']) / self.grid_params['dy'])
            )
            # Exclusive index (upper) of destination grid point above the top-most
            # influence of radar data
            yi_max = min(
                self.grid_params['ny'],
                int(
                    (y_radar + r_max - self.grid_params['y_min'])
                    / self.grid_params['dy']
                ) + 2
            )
            # Sanity check that we didn't exceed our bounds
            if (
                xi_max <= 0
                or xi_min >= self.grid_params['nx']
                or yi_max <= 0
                or yi_min >= self.grid_params['ny']
            ):
                warnings.warn("Radar included outside of maximum range box. Skipping.")
                continue

            # Prep coordinates of gates on destination grid
            site_id = radar.metadata['instrument_name']
            gate_dest_xy = radar_coords_to_grid_coords(
                radar.gate_x['data'],
                radar.gate_y['data'],
                site_id=site_id,
                transformer=transformer,
                wait_for_cache=(site_id in radar_site_ids)
            )
            gate_dest_z = radar.gate_altitude['data']
            
            radar_site_ids.append(site_id)
            radars.append({
                'radar': radar,
                'x_radar': x_radar,
                'y_radar': y_radar,
                'transformer': transformer,
                'xi_min': xi_min,
                'xi_max': xi_max,
                'yi_min': yi_min,
                'yi_max': yi_max,
                'gate_dest_xy': gate_dest_xy,
                'gate_dest_z': gate_dest_z
            })

        def compute_coords(self):
            """Run dask.compute on horizontal coordinates of radar gates."""
            coords = dask.compute(*tuple(radar['gate_dest_xy'] for radar in self.radars))
            for i, radar in enumerate(self.radars):
                radar['gate_dest_x'] = coords[i][0]
                radar['gate_dest_y'] = coords[i][1]
                del radar['gate_dest_xy']

        def map_gates_to_grid(
            self,
            fields,
            analysis_time=None,
            weighting_function='GridRad',
            gatefilters=False,
            map_roi=False,
            roi_func='dist_beam',
            constant_roi=None,
            z_factor=0.05,
            xy_factor=0.02,
            min_radius=500.0,
            h_factor=1.0,
            nb=1.5,
            bsp=1.0,
            filter_kwargs=None
        ):
            """Run the actual regridding using Py-ART style routine.

            TODO: document
            """

            self.analysis_time = analysis_time

            filter_kwargs = {} if filter_kwargs is None else filter_kwargs
            gatefilters = _parse_gatefilters(
                gatefilters,
                [radar['radar'] for radar in radars]
            )
            cy_weighting_function = _determine_cy_weighting_func(weighting_function)
            fields = _determine_fields(fields, [radar['radar'] for radar in radars])
            offsets = [
                (radar['radar'].altitude['data'].item(), radar['y_radar'], radar['x_radar'])
                for radar in radars
            ]
            roi_func = _parse_roi_func(
                roi_func,
                constant_roi,
                z_factor,
                xy_factor,
                min_radius,
                h_factor,
                nb,
                bsp,
                offsets
            )

            nfields = len(fields)

            subgrids = []
            for radar, gatefilter in zip(self.radars, gatefilters):
                subgrid_shape = (
                    self.grid_params['nz'],
                    radar['yi_max'] - radar['yi_min'],
                    radar['xi_max'] - radar['xi_min'],
                    nfields
                )
                subgrid_starts = None  # TODO
                subgrid_steps = None  # TODO

                # Copy field data and masks
                field_shape = (radar['radar'].nrays, radar['radar'].ngates, nfields)
                field_data = np.empty(field_shape, dtype='float32')
                field_mask = np.empty(field_shape, dtype='float32')
                for i, field in enumerate(fields):
                    fdata = radar['radar'].fields[field]['data']
                    field_data[:, :, i] = np.ma.getdata(fdata)
                    field_mask[:, :, i] = np.ma.getmaskarray(fdata)

                # Find excluded gates from gatefilter
                if gatefilter is False:
                    gatefilter = GateFilter(radar['radar'])
                elif gatefilter is None:
                    gatefilter = moment_based_gate_filter(radar['radar'], **filter_kwargs)
                excluded_gates = gatefilter.gate_excluded.astype('uint8')

                # Range and offsets
                gate_range = radar['radar'].range['data'].astype('float32')
                if analysis_time is None:
                    gate_timedelta = np.zeros(radar['radar'].nrays, dtype='float32')
                else:
                    gate_timedelta = np.array([
                        (t - analysis_time.replace(tzinfo=None)).total_seconds()
                        for t in num2date(radar['radar'].time['data'], radar['radar'].time['units'])
                    ], dtype='float32')

                # Delayed computation of mapping of this radar's gates to its subgrid values
                # and weights
                subgrids.append(map_gates_to_subgrid(
                    subgrid_shape,
                    subgrid_starts,
                    subgrid_steps,
                    field_shape,
                    field_data,
                    field_mask,
                    excluded_gates,
                    radar['gate_dest_z'],
                    radar['gate_dest_y'],
                    radar['gate_dest_x'],
                    gate_range,
                    gate_timedelta,
                    self.grid_params['z_max'],
                    roi_func,
                    cy_weighting_function
                ))
                
            # Run the computation on the subgrids
            subgrids = tuple(dask.compute(*subgrids))

            # Sum the subgrids
            grid_shape = (self.grid_params['nz'], self.grid_params['ny'], self.grid_params['nx'], nfields)
            grid_sum = np.zeros(grid_shape, dtype='float32')
            grid_wsum = np.zeros(grid_shape, dtype='float32')
            for subgrid, radar in zip(subgrids, radars):
                grid_sum[:, radar['yi_min']:radar['yi_max'], radar['xi_min']:radar['xi_max'], :] += subgrid['sum']
                grid_wsum[:, radar['yi_min']:radar['yi_max'], radar['xi_min']:radar['xi_max'], :] += subgrid['wsum']

            # Apply the weighting and masking
            mweight = np.ma.masked_equal(grid_wsum, 0)
            msum = np.ma.masked_array(grid_sum, mweight.mask)
            self.grids = dict(
                (f, msum[..., i] / mweight[..., i]) for i, f in enumerate(fields)
            )
            
            # NOTE: Py-ART's map_roi option is ignored due to difference in gatemapper construction
            return self.grids

        def to_xarray(self):
            """Return saved grids as CF-compliant xarray Dataset."""
            ds = generate_rectangular_grid(
                self.grid_params['nx'],
                self.grid_params['ny'],
                self.grid_params['dx'],
                self.grid_params['dy'],
                self.cf_attrs,
                x0=self.grid_params['x_min'],
                y0=self.grid_params['y_min']
            )
            ds = ds.assign_coords(
                z=xr.DataArray(
                    np.arange(self.grid_params['nz']) * self.grid_params['dz'] + self.grid_params['z_min'],
                    dims=('z',),
                    name='z',
                    attrs={
                        'standard_name': 'altitude',
                        'units': 'meter',
                        'positive': 'up'
                    }
                ),
                time=xr.DataArray(
                    self.analysis_Time,
                    name='time',
                    attrs={
                        'long_name': 'Time of radar analysis'
                    }
                )
            )
            ds = ds.assign(
                {
                    field: xr.Variable(
                        ('z', 'y', 'x'),
                        self.grids[field],
                        {k: v for k, v in self.radars[0]['radar'].fields[field].items() if k in ['units', 'standard_name', 'long_name']}
                    )
                    for field in self.grids
                }
            )

            return ds

