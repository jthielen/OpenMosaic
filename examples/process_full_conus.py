#!/usr/bin/env python
# Copyright (c) 2020 OpenMosaic Developers.
# Distributed under the terms of the Apache 2.0 License
# SPDX-License-Identifier: Apache-2.0

import argparse
from itertools import product, repeat
import logging
import os
import re
from sqlite3 import dbapi2 as sql
import warnings

from dask.distributed import Client, LocalCluster
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

import pyproj
import boto3
import pyart
import cftime
import shapely

from openmosaic.gridding import Gridder, generate_rectangular_grid
from openmosaic.utils import get_nexrad_sites


# Initial parameters
log = logging.getLogger(__name__)
bucket_name = 'noaa-nexrad-level2'
s3 = boto3.client('s3')
l2_datetime_pattern = re.compile(
    r"(?:K[A-Z]{3})(?P<Y>[0-9]{4})(?P<m>[0-9]{2})(?P<d>[0-9]{2})_(?P<H>[0-9]{2})"
    r"(?P<M>[0-9]{2})(?P<S>[0-9]{2})"
)

# Function definitions
def aggregate_to_dataset(fields, grid_params, cf_attrs, analysis_time, *subgrid_data_and_weights):
    """Aggregate results from individual radar subgrids onto main grid."""
    grid_shape = (grid_params['nz'], grid_params['ny'], grid_params['nx'], len(fields))
    grid_sum = np.zeros(grid_shape, dtype='float32')
    grid_wsum = np.zeros(grid_shape, dtype='float32')
    field_metadata = {field: '' for field in fields}  # If no valid data, have to mock metadata values
    for subgrid in subgrid_data_and_weights:
        if subgrid is not None:
            field_metadata = subgrid['field_metadata']
            grid_sum[:, subgrid['yi_min']:subgrid['yi_max'], subgrid['xi_min']:subgrid['xi_max'], :] += subgrid['sum']
            grid_wsum[:, subgrid['yi_min']:subgrid['yi_max'], subgrid['xi_min']:subgrid['xi_max'], :] += subgrid['wsum']

    # Apply the weighting and masking
    mweight = np.ma.masked_equal(grid_wsum, 0)
    msum = np.ma.masked_array(grid_sum, mweight.mask)
    grids = {f: (msum[..., i] / mweight[..., i]) for i, f in enumerate(fields)}

    # Create Dataset from dictionary
    ds = generate_rectangular_grid(
        grid_params['nx'],
        grid_params['ny'],
        grid_params['dx'],
        grid_params['dy'],
        cf_attrs,
        x0=grid_params['x_min'],
        y0=grid_params['y_min']
    )
    ds = ds.assign_coords(
        z=xr.DataArray(
            np.arange(grid_params['nz']) * grid_params['dz'] + grid_params['z_min'],
            dims=('z',),
            name='z',
            attrs={
                'standard_name': 'altitude',
                'units': 'meter',
                'positive': 'up'
            }
        ),
        time=xr.DataArray(
            analysis_time,
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
                grids[field],
                field_metadata[field]
            )
            for field in grids
        }
    )

    return ds


def post_ops(dataset):
    """Post subbatch dataset operations to go from 3D to 2D.
    
    Notes
    -----
    Currently only supports column-maximum reflectivity calculations!
    """
    return dataset.max('z').drop_vars(['lon', 'lat'])


def download_nexrad_file(file_key, bucket_name='noaa-nexrad-level2'):
    s3 = boto3.client('s3')
    f = temp_l2_dir + file_key.split("/")[-1]
    if not os.path.isfile(f):
        log.info(f"Downloading {f}")
        s3.download_file(bucket_name, file_key, f)
    return f


if __name__ == '__main__':
    # Set up script configuration
    parser = argparse.ArgumentParser(
        description="Create single dataset of CONUS mosaics at single time.", usage=argparse.SUPPRESS
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        help="<Required> Output Directory",
        required=True,
        metavar="~/output/",
    )
    parser.add_argument(
        "-t",
        "--timestamp",
        help="<Required> UTC Timestamp(s), space-separated",
        required=True,
        metavar="2015-07-03T01:00Z",
    )
    parser.add_argument(
        "-c", "--l2-dir", help="Level II cache directory", default="~/l2_cache/", metavar="~/l2_cache/"
    )
    parser.add_argument(
        "-C", "--reg-dir", help="Regression cache directory", default="~/reg_cache/", metavar="~/reg_cache/"
    )
    parser.add_argument(
        "-n",
        "--nexrad-file",
        help="NEXRAD Site GeoJSON file",
        default="~/Weather_Radar_Stations.geojson",
        metavar="~/Weather_Radar_Stations.geojson"
    )
    parser.add_argument(
        "-R", "--r-max", help="Max data radius in meters", default="3.0e5", metavar="3.0e5"
    )
    parser.add_argument(
        "-V", "--vol-search", help="Volume search interval (sec)", default="600", metavar="600"
    )
    parser.add_argument(
        "-S", "--sweep-search", help="Sweep search interval (sec)", default="228", metavar="228"
    )

    try:
        args = parser.parse_args()
        output_dir = args.output_dir
        timestamps = [pd.Timestamp(ts) for ts in args.timestamp.split(' ') if ts]
        r_max = float(args.r_max)  # (meters) max distance for which radar data is included
        vol_search_interval = np.timedelta64(int(args.vol_search), 's')  # time window for which to search for useful sweeps
        sweep_interval = np.timedelta64(int(args.sweep_search), 's')  # time window for data inclusion
        temp_l2_dir = args.l2_dir
        temp_reg_dir = args.reg_dir
        nexrad_file = args.nexrad_file
    except (SystemExit, ValueError):
        parser.print_help()
        raise

    def maybe_float(val):
        try:
            return float(val)
        except:
            return val

    # Hard-code CF attrs for now...
    cf_attrs = {
        'grid_mapping_name': 'lambert_conformal_conic',
        'latitude_of_projection_origin': 38.5,
        'longitude_of_central_meridian': 262.5,
        'standard_parallel': 38.5,
        'earth_radius': 6371229.0
    }
    crs = pyproj.CRS.from_cf(cf_attrs)
    proj = pyproj.Proj(crs)
    
    # Output control parameters for reference

    print(f"Timestamp(s):")
    for timestamp in timestamps:
        print(timestamp.strftime("\t%Y-%m-%d %H:%M"))
    print("Projection Used:")
    for k, v in cf_attrs.items():
        print(f"\t{k}: {v}")
    print(f"Radar Cache dir: {temp_l2_dir}")
    print(f"Regression Cache dir: {temp_reg_dir}")
    print(f"NEXRAD GeoJSON file: {nexrad_file}")
    print(f"Output dir: {output_dir}")
    print(f"r_max: {r_max}")
    print(f"vol_search_interval: {vol_search_interval.item().seconds}")
    print(f"sweep_interval: {sweep_interval.item().seconds}")

    # Set up dask
    cluster = LocalCluster()
    cluster.scale(6)
    client = Client(cluster)
    print(client)

    # Init records to base processing off of
    radar_sites = get_nexrad_sites(nexrad_file, crs)
    radar_sites['x'] = radar_sites['proj_geom'].apply(lambda point: point.x)
    radar_sites['y'] = radar_sites['proj_geom'].apply(lambda point: point.y)

    # Define horizontal bounds
    horizontal_params = {
        'x_min': -2.3e6,
        'x_max': 2.4e6,
        'y_min': -1.5e6,
        'y_max': 1.5e6
    }
    # Subset the radar definition file to only include those sites in your domain
    radar_search_area = shapely.geometry.box(
        horizontal_params['x_min'] - r_max,
        horizontal_params['y_min'] - r_max,
        horizontal_params['x_max'] + r_max,
        horizontal_params['y_max'] + r_max
    )
    radars_to_use = radar_sites[[radar_search_area.contains(p) for p in radar_sites['proj_geom']]]['siteID']

    # Shift analysis times forward and back to get the dates we have to search S3 for
    dates_to_search = np.unique(np.concatenate([
        [(timestamp - vol_search_interval).floor('D') for timestamp in timestamps],
        [(timestamp + vol_search_interval).floor('D') for timestamp in timestamps]
    ]))

    # Build list of radar files/S3 keys to use for this batch
    # Slow process, as not easily dask-parallelizable due to projections
    print("\nQuerying S3 for radar files...\n")

    # First, since S3 search is organized by date and site, collect those lists all at once to
    # save querying time
    s3_key_lists = {}
    for date in dates_to_search:
        for radar_id in radars_to_use:
            s3_search = s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix=date.strftime(f"%Y/%m/%d/{radar_id}")
            )
            if 'Contents' not in s3_search:
                warnings.warn(date.strftime(f"No files found for {radar_id} on %Y-%m-%d"))
                continue
            s3_key_lists[(date, radar_id)] = [
                obj['Key'] for obj in s3_search['Contents']
            ]

    # Now, filter the lists for just the radars and times needed
    file_keys = []
    for i, timestamp in enumerate(timestamps):
        analysis_time = timestamp.tz_convert(None)
        for date, radar_id in product(dates_to_search, radars_to_use):
            try:
                for f in s3_key_lists[(date, radar_id)]:
                    match = l2_datetime_pattern.search(f)
                    if match and (
                        analysis_time - vol_search_interval
                        <= pd.Timestamp("{Y}-{m}-{d}T{H}:{M}:{S}".format(**match.groupdict()))
                        <= analysis_time + vol_search_interval
                    ):
                        file_keys.append((i, radar_id, f))
            except KeyError:
                # None found!
                continue
    
    # Submit downloads to client
    nexrad_files_all = client.map(download_nexrad_file, [tup[2] for tup in file_keys], priority=1)

    ######################
    # Main Subbatch Loop #
    ######################
    print("\nStarting primary time loop...see Dask dashboard...\n")
    datasets = []
    radar_sites_no_geo = pd.DataFrame(radar_sites[['x', 'y']])
    for i, timestamp in enumerate(timestamps):
        # Create gridder
        g = Gridder(cf_attrs, nz=24, dz=1000., z_min=1000.)
        g.assign_from_subbatch_and_spacing(horizontal_params, 2000.)
        g.cache_dir = temp_reg_dir

        # Filter files for this time
        nexrad_files, nexrad_site_ids = zip(*[
            (nexrad_path_future, file_keys[j][1])
            for j, nexrad_path_future in enumerate(nexrad_files_all)
            if file_keys[j][0] == i
        ])

        # Predefine radar subgrids
        g.prepare_radar_subgrids(set(nexrad_site_ids), radar_sites_no_geo, r_max)

        # Go from nexrad file on disk to 4D subgrid at analysis time (extra dim is fields)
        fields = ['reflectivity']
        subgrid_data_and_weights_futures = client.map(
            g.process_nexrad_file_to_subgrids_and_weights,
            nexrad_files,
            nexrad_site_ids,
            fields=fields,
            analysis_time=timestamp.tz_convert(None),
            sweep_interval=sweep_interval,
            priority=2
        )

        # Go from collection of subgrid data and weights to full 3D xarray dataset
        # def aggregate_to_dataset(*subgrid_data_and_weights)
        dataset_full = client.submit(
            aggregate_to_dataset,
            fields,
            g.grid_params,
            g.cf_attrs,
            timestamp.tz_convert(None),
            *subgrid_data_and_weights_futures,
            priority=3
        )

        # def post_ops(dataset) (such as flattening, etc.)
        datasets.append(client.submit(post_ops, dataset_full, priority=4))

    gathered_datasets = client.gather(datasets)
    output = xr.concat(gathered_datasets, dim='time')
    
    # Clean up output
    output.reflectivity.data[output.reflectivity.data <= 0] = np.nan
    output['projection'] = gathered_datasets[0].projection
    output.reflectivity.attrs = {
        'units': 'dBZ',
        'long_name': 'column maximum reflectivity'
    }
        
    output.to_netcdf(
        output_dir
        + "openmosaic_conus_"
        + timestamps[0].strftime("%Y%m%dT%H%M%S")
        + ("_to_" + timestamps[0].strftime("%Y%m%dT%H%M%S") + f"_{len(timestamps)}_times" if len(timestamps) > 1 else "")
        + ".nc"
    )

    print(f"\n\nSUCCESS: PROCESSED TIMESTAMPS {timestamps}")
