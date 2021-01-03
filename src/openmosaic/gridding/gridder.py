"""
Gridder class (and directly supporting utils) for efficient and parallelized mapping of
Py-ART Radar objects to a common, regular grid. Supported by the underlying subgrid mapping
of the GatesToSubgridMapper.

Portions inspired by the map_gates_to_grid implementation of Py-ART (license reproduced in
_map_gates_to_subgrid.pyx, which more directly reuses Py-ART code).

"""

from time import sleep
import logging
from pathlib import Path
from joblib import dump, load
import warnings

from cftime import num2date
import numpy as np
import pyart
from pyart.core.radar import Radar
from pyart.filters import GateFilter, moment_based_gate_filter
import pyproj
from sklearn.linear_model import ElasticNet
import xarray as xr

from ._map_gates_to_subgrid import GatesToSubgridMapper
from .grid_utils import generate_rectangular_grid
from .vendored import (
    _determine_cy_weighting_func,
    _parse_gatefilters,
    _determine_fields,
    _parse_roi_func
)


log = logging.getLogger(__name__)

_grid_param_labels_z = ['nz', 'dz', 'z_min', 'z_max']
_grid_param_labels_y = ['ny', 'dy', 'y_min', 'y_max']
_grid_param_labels_x = ['nx', 'dx', 'x_min', 'x_max']


def radar_coords_to_grid_coords(
    gate_radar_x,
    gate_radar_y,
    site_id,
    radar_crs_kwargs,
    target_crs_cf_attrs,
    wait_for_cache=False,
    cache_dir='/tmp/'
):
    """Map radar x/y to grid x/y using a quadratic model of projection transform.
    
    Uses an ElasticNet regression over 2D quadratic terms to transform coordinates from radar
    azimuthal equidistant space to common grid. If the cached regression for a particular
    radar site is not available, perform the slow transform with pyproj in order to train the
    regression for later use.

    This technique can give a more than order of magnitude speed up, with errors less than
    100 m within 300 km of the radar site (also, faster and signficantly less error than
    using pyproj.Proj from Py-ART's calcuated lat/lons).

    WARNING: If you change the destination grid, you must clear the cached regressions in the
    .regressions directory.
    """
    # Ensure float32
    gate_x = gate_radar_x.astype('float32')
    gate_y = gate_radar_y.astype('float32')
    gate_x_r = gate_x.ravel()
    gate_y_r = gate_y.ravel()

    # Check for cached version
    regression_path = Path(cache_dir) / "./regressions"
    regression_path.mkdir(parents=True, exist_ok=True)
    use_saved = False
    x_regression_path = regression_path / f"{site_id}_reg_x.joblib"
    y_regression_path = regression_path / f"{site_id}_reg_y.joblib"

    if x_regression_path.exists() and y_regression_path.exists():
        x_clf = load(x_regression_path)
        y_clf = load(y_regression_path)
        use_saved = True
    elif wait_for_cache:
        # Wait 5 seconds for cached version
        sleep(5)
        if x_regression_path.exists() and y_regression_path.exists():
            x_clf = load(x_regression_path)
            y_clf = load(y_regression_path)
            save_to_cache = True

    if not use_saved:
        # Create the transformed coordinate, and train the regression
        # (slices every 11 since subset performs better, and 11 not factor of 360 to avoid
        # periodicity)
        radar_crs = pyproj.CRS(radar_crs_kwargs)
        target_crs = pyproj.CRS.from_cf(target_crs_cf_attrs)
        transformer = pyproj.Transformer.from_crs(radar_crs, target_crs)
        grid_x, grid_y = transformer.transform(gate_x, gate_y)
        x_clf = ElasticNet().fit(np.stack([
            gate_x_r,
            gate_y_r,
            gate_x_r**2,
            gate_y_r**2,
            gate_x_r * gate_y_r
        ], axis=1)[::11], grid_x.ravel()[::11])
        y_clf = ElasticNet().fit(np.stack([
            gate_x_r,
            gate_y_r,
            gate_x_r**2,
            gate_y_r**2,
            gate_x_r * gate_y_r
        ], axis=1)[::11], grid_y.ravel()[::11])
        dump(x_clf, x_regression_path)
        dump(y_clf, y_regression_path)

    # Use the 2D quadratic model of proj transform
    grid_x = x_clf.predict(np.stack([
        gate_x_r,
        gate_y_r,
        gate_x_r**2,
        gate_y_r**2,
        gate_x_r * gate_y_r
    ], axis=1)).reshape(gate_x.shape)
    grid_y = y_clf.predict(np.stack([
        gate_x_r,
        gate_y_r,
        gate_x_r**2,
        gate_y_r**2,
        gate_x_r * gate_y_r
    ], axis=1)).reshape(gate_y.shape)

    return np.stack([grid_x, grid_y], axis=0)


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
    roi_func_args,
    cy_weighting_function
):
    """Delayed functional wrapper of GatesToSubgridMapper class."""
    subgrid_sum = np.zeros(subgrid_shape, dtype=np.float32)
    subgrid_wsum = np.zeros(subgrid_shape, dtype=np.float32)
    gatemapper = GatesToSubgridMapper(
        subgrid_shape[:-1],
        subgrid_starts,
        subgrid_steps,
        subgrid_sum,
        subgrid_wsum
    )

    roi_func = _parse_roi_func(*roi_func_args)

    gatemapper.map_gates_to_subgrid(
        field_shape[1],
        field_shape[0],
        gate_z.astype('float32'),
        gate_y.astype('float32'),
        gate_x.astype('float32'),
        gate_range.astype('float32'),
        gate_timedelta.astype('float32'),
        field_data,
        field_mask,
        excluded_gates,
        toa,
        roi_func,
        cy_weighting_function
    )

    return {
        'sum': subgrid_sum,
        'wsum': subgrid_wsum
    }


def prepare_single_radar(i, radar, radar_coords, grid_params, r_max, cf_attrs, cache_dir):
    # Define projection objects
    crs_kwargs = {
        'proj': 'aeqd',
        'lat_0': radar.latitude['data'].item(),
        'lon_0': radar.longitude['data'].item()
    }

    # Determine subset of grid that this radar will map data on to
    x_radar, y_radar = radar_coords[i]
    # Inclusive index of destination grid point to the left of left-most influence of
    # radar data
    xi_min = max(
        0,
        int((x_radar - r_max - grid_params['x_min']) / grid_params['dx'])
    )
    # Exclusive index (upper) of destination grid point to the right of the right-most
    # influence of radar data
    xi_max = min(
        grid_params['nx'],
        int(
            (x_radar + r_max - grid_params['x_min'])
            / grid_params['dx']
        ) + 2
    )
    # Inclusive index of destination grid point below the bottom-most influence of
    # radar data
    yi_min = max(
        0,
        int((y_radar - r_max - grid_params['y_min']) / grid_params['dy'])
    )
    # Exclusive index (upper) of destination grid point above the top-most
    # influence of radar data
    yi_max = min(
        grid_params['ny'],
        int(
            (y_radar + r_max - grid_params['y_min'])
            / grid_params['dy']
        ) + 2
    )
    # Sanity check that we didn't exceed our bounds
    if (
        xi_max <= 0
        or xi_min >= grid_params['nx']
        or yi_max <= 0
        or yi_min >= grid_params['ny']
    ):
        warnings.warn("Radar included outside of maximum range box. Skipping.")
        return

    # Prep coordinates of gates on destination grid
    site_id = radar.metadata['instrument_name']
    gate_dest_xy = radar_coords_to_grid_coords(
        radar.gate_x['data'],
        radar.gate_y['data'],
        site_id=site_id,
        radar_crs_kwargs=crs_kwargs,
        target_crs_cf_attrs=cf_attrs,
        wait_for_cache=False,  # TODO validate assumption
        cache_dir=cache_dir
    )
    gate_dest_z = radar.gate_altitude['data']
    
    return {
        'site_id': site_id,
        'radar': radar,
        'x_radar': x_radar,
        'y_radar': y_radar,
        'xi_min': xi_min,
        'xi_max': xi_max,
        'yi_min': yi_min,
        'yi_max': yi_max,
        'gate_dest_x': gate_dest_xy[0],
        'gate_dest_y': gate_dest_xy[1],
        'gate_dest_z': gate_dest_z
    }

class Gridder:
    """Map gates from one or more radars to a common grid.

    TODO: document

    """

    def __init__(self, cf_projection, **grid_params):
        """Set up Gridder by defining grid.

        Parameters
        ----------
        cf_projection : dict-like
            Dictionary of projection parameters as defined by the CF Conventions
        pool : multiprocessing.Pool
            Pool for multiprocessing parallelization where useful
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
        self.cache_dir = "/tmp/"
        self.grid_params = {}
        self.r_max = 3.0e5

        self._assign_grid_params('x', {k: v for k, v in grid_params.items() if k in _grid_param_labels_x})
        self._assign_grid_params('y', {k: v for k, v in grid_params.items() if k in _grid_param_labels_y})
        self._assign_grid_params('z', {k: v for k, v in grid_params.items() if k in _grid_param_labels_z})
        
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
            self.round_grid_params(
                {'x_min': subbatch['x_min'], 'x_max': subbatch['x_max'], 'dx': spacing},
                dims='x'
            )
        )
        self._assign_grid_params(
            'y',
            self.round_grid_params(
                {'y_min': subbatch['y_min'], 'y_max': subbatch['y_max'], 'dy': spacing},
                dims='y'
            )
        )

    def _assign_grid_params(self, dim, params):
        """Assign grid parameters for dimension."""
        if not params:
            # Skip if empty
            return

        n = params.get(f"n{dim}", None)
        d = params.get(f"d{dim}", None)
        min = params.get(f"{dim}_min", None)
        max = params.get(f"{dim}_max", None)

        if len([param for param in [n, d, min, max] if param is None]) != 1:
            warnings.warn(str((n, d, min, max)))
            raise ValueError(
                f"Exactly three of four grid parameters must be specified for a "
                f"well-defined grid for dimension {dim}."
            )

        if n is None:
            n = int((max - min) / d) + 1
            if np.abs(max - min - d * (n - 1)) > 1e-6:
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

    @staticmethod
    def round_grid_params(grid_params, dims=None):
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
            grid_params[f"{dim}_min"] = (
                np.floor(grid_params[f"{dim}_min"] / grid_params[f"d{dim}"])
                * grid_params[f"d{dim}"]
            )
            grid_params[f"{dim}_max"] = (
                np.ceil(grid_params[f"{dim}_max"] / grid_params[f"d{dim}"])
                * grid_params[f"d{dim}"]
            )
            if f"n{dim}" in grid_params:
                grid_params[f"n{dim}"] = int(
                    (grid_params[f"{dim}_max"] - grid_params[f"{dim}_min"])
                    / grid_params[f"d{dim}"]
                ) + 1

        return grid_params

    def prepare_radar_subgrids(self, radar_site_ids, radar_sites, r_max=3.0e5):
        self.subgrid_params = {}
        for radar_id in radar_site_ids:
            # Get radar location
            radar_info = radar_sites.loc[radar_id]
            radar_location_x = radar_info['x']
            radar_location_y = radar_info['y']

            # Prepare naive subgrid for this radar
            subgrid_params = self.round_grid_params({
                'x_min': radar_location_x - r_max,
                'x_max': radar_location_x + r_max,
                'dx': self.grid_params['dx'],
                'nx': None,
                'y_min': radar_location_y - r_max,
                'y_max': radar_location_y + r_max,
                'dy': self.grid_params['dy'],
                'ny': None,
                'z_min': self.grid_params['z_min'],
                'z_max': self.grid_params['z_max'],
                'dz': self.grid_params['dz'],
                'nz': None
            })

            # Intersect with overall grid
            subgrid_params['x_min'] = max(subgrid_params['x_min'], self.grid_params['x_min'])
            subgrid_params['x_max'] = min(subgrid_params['x_max'], self.grid_params['x_max'])
            subgrid_params['y_min'] = max(subgrid_params['y_min'], self.grid_params['y_min'])
            subgrid_params['y_max'] = min(subgrid_params['y_max'], self.grid_params['y_max'])

            # Go back and compute indexes
            subgrid_params['xi_min'] = int(
                (subgrid_params['x_min'] - self.grid_params['x_min']) / self.grid_params['dx']
            )
            subgrid_params['xi_max'] = int(
                (subgrid_params['x_max'] - self.grid_params['x_min']) / self.grid_params['dx']
            ) + 1
            subgrid_params['nx'] = subgrid_params['xi_max'] - subgrid_params['xi_min']
            subgrid_params['yi_min'] = int(
                (subgrid_params['y_min'] - self.grid_params['y_min']) / self.grid_params['dy']
            )
            subgrid_params['yi_max'] = int(
                (subgrid_params['y_max'] - self.grid_params['y_min']) / self.grid_params['dy']
            ) + 1
            subgrid_params['ny'] = subgrid_params['yi_max'] - subgrid_params['yi_min']

            # Save other info
            subgrid_params['x_radar'] = radar_location_x
            subgrid_params['y_radar'] = radar_location_y

            # Save result
            self.subgrid_params[radar_id] = subgrid_params

    def process_nexrad_file_to_subgrids_and_weights(
        self,
        nexrad_file_path,
        radar_id,
        fields,
        sweep_interval,
        analysis_time=None,
        weighting_function='GridRad',
        gatefilter=False,
        map_roi=False,
        roi_func='dist_beam',
        constant_roi=None,
        z_factor=0.05,
        xy_factor=0.02,
        min_radius=500.0,
        h_factor=1.0,
        nb=1.5,
        bsp=1.0,
        filter_kwargs=None,
        cache_dir=None
    ):
        """Do basically everything for a single radar file.

        Read the file, filter for sweeps, calcuate coordinates, and map to subgrid.
        
        PARAMETERS TODO.
        """
        ##############################
        # Read and filter radar file #
        ##############################
        try:
            radar = pyart.io.read_nexrad_archive(nexrad_file_path)
        except:
            warnings.warn(f"Cannot read file {nexrad_file_path}")
            return None

        sweep_time_offsets = [np.median(radar.time['data'][s:e]) for s, e in radar.iter_start_end()]
        sweep_times = num2date(sweep_time_offsets, radar.time['units'])
        valid_sweep_ids = [
            i for i, t in enumerate(sweep_times) if (
                analysis_time - sweep_interval
                <= t
                <= analysis_time + sweep_interval
            )
        ]
        if valid_sweep_ids:
            radar = radar.extract_sweeps(valid_sweep_ids)
        else:
            log.info(f"File {nexrad_file_path} does not contain valid sweep times")
            del radar
            return None

        #############################
        # Compute Radar Coordinates #
        #############################
        crs_kwargs = {
            'proj': 'aeqd',
            'lat_0': radar.latitude['data'].item(),
            'lon_0': radar.longitude['data'].item()
        }
        gate_dest_x, gate_dest_y = radar_coords_to_grid_coords(
            radar.gate_x['data'],
            radar.gate_y['data'],
            site_id=radar_id,
            radar_crs_kwargs=crs_kwargs,
            target_crs_cf_attrs=self.cf_attrs,
            wait_for_cache=False,  # TODO validate assumption
            cache_dir=self.cache_dir
        )
        gate_dest_z = radar.gate_altitude['data']

        ##################
        # Map to Subgrid #
        ##################
        
        # Arguments
        filter_kwargs = {} if filter_kwargs is None else filter_kwargs
        cy_weighting_function = _determine_cy_weighting_func(weighting_function)
        offsets = [(
            radar.altitude['data'].item(),
            self.subgrid_params[radar_id]['y_radar'],
            self.subgrid_params[radar_id]['x_radar']
        )]
        roi_func_args = (
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

        # Subgrid setup
        subgrid_shape = (
            self.grid_params['nz'],
            self.subgrid_params[radar_id]['ny'],
            self.subgrid_params[radar_id]['nx'],
            nfields
        )
        subgrid_starts = (
            self.grid_params['z_min'],
            self.subgrid_params[radar_id]['y_min'],
            self.subgrid_params[radar_id]['x_min']
        )
        subgrid_steps = (
            self.grid_params['dz'],
            self.grid_params['dy'],
            self.grid_params['dx'],
        ) 

        # Copy field data and masks
        field_shape = (radar.nrays, radar.ngates, nfields)
        field_data = np.empty(field_shape, dtype='float32')
        field_mask = np.empty(field_shape, dtype='uint8')
        for i, field in enumerate(fields):
            fdata = radar.fields[field]['data']
            field_data[:, :, i] = np.ma.getdata(fdata)
            field_mask[:, :, i] = np.ma.getmaskarray(fdata)

        # Find excluded gates from gatefilter
        if gatefilter is False:
            gatefilter = GateFilter(radar)
        elif gatefilter is None:
            gatefilter = moment_based_gate_filter(radar, **filter_kwargs)
        excluded_gates = gatefilter.gate_excluded.astype('uint8')

        # Range and offsets
        gate_range = radar.range['data'].astype('float32')
        if analysis_time is None:
            gate_timedelta = np.zeros(radar.nrays, dtype='float32')
        else:
            gate_timedelta = np.array([
                (t - analysis_time.replace(tzinfo=None)).total_seconds()
                for t in num2date(radar.time['data'], radar.time['units'])
            ], dtype='float32')

        # Apply low-level map_gates_to_subgrid
        subgrid = map_gates_to_subgrid(
            subgrid_shape,
            subgrid_starts,
            subgrid_steps,
            field_shape,
            field_data,
            field_mask,
            excluded_gates,
            gate_dest_z,
            gate_dest_y,
            gate_dest_x,
            gate_range,
            gate_timedelta,
            self.grid_params['z_max'],
            roi_func_args,
            cy_weighting_function
        )

        # Inject subgrid info and return
        return {
            **subgrid,
            **self.subgrid_params[radar_id],
            'field_metadata': {
                field: {
                    k: v for k, v in radar.fields[field].items() if k in ['units', 'standard_name', 'long_name']
                } for field in fields
            }
        }

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
            [radar['radar'] for radar in self.radars]
        )
        cy_weighting_function = _determine_cy_weighting_func(weighting_function)
        fields = _determine_fields(fields, [radar['radar'] for radar in self.radars])
        offsets = [
            (radar['radar'].altitude['data'].item(), radar['y_radar'], radar['x_radar'])
            for radar in self.radars
        ]
        roi_func_args = (
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

        subgrid_args = []
        for radar, gatefilter in zip(self.radars, gatefilters):
            subgrid_shape = (
                self.grid_params['nz'],
                radar['yi_max'] - radar['yi_min'],
                radar['xi_max'] - radar['xi_min'],
                nfields
            )
            subgrid_starts = (
                self.grid_params['z_min'],
                self.grid_params['y_min'] + self.grid_params['dy'] * radar['yi_min'],
                self.grid_params['x_min'] + self.grid_params['dx'] * radar['xi_min']
            )
            subgrid_steps = (
                self.grid_params['dz'],
                self.grid_params['dy'],
                self.grid_params['dx'],
            ) 

            # Copy field data and masks
            field_shape = (radar['radar'].nrays, radar['radar'].ngates, nfields)
            field_data = np.empty(field_shape, dtype='float32')
            field_mask = np.empty(field_shape, dtype='uint8')
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

            # Add args to list to be applied to map_gates_to_subgrid
            subgrid_args.append((
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
                roi_func_args,
                cy_weighting_function
            ))
            
        # Run the computation
        if self.pool is None:
            subgrids = []
            for args in subgrid_args:
                subgrids.append(map_gates_to_subgrid(*args))
        else:
            subgrids = list(self.pool.starmap(map_gates_to_subgrid, subgrid_args))

        # Sum the subgrids
        grid_shape = (self.grid_params['nz'], self.grid_params['ny'], self.grid_params['nx'], nfields)
        grid_sum = np.zeros(grid_shape, dtype='float32')
        grid_wsum = np.zeros(grid_shape, dtype='float32')
        for subgrid, radar in zip(subgrids, self.radars):
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
                self.analysis_time,
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

