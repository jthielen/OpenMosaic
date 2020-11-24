# Copyright (c) 2020 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Generic shared utils."""

import logging
import os
import re
import warnings

import boto3
import cftime
import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import pyart
import shapely
from tqdm import tqdm


log = logging.getLogger(__name__)
l2_datetime_pattern = re.compile(
    r"(?:K[A-Z]{3})(?P<Y>[0-9]{4})(?P<m>[0-9]{2})(?P<d>[0-9]{2})_(?P<H>[0-9]{2})"
    r"(?P<M>[0-9]{2})(?P<S>[0-9]{2})"
)


def get_nexrad_sites(geojson_file, crs):
    """Obtain GeoDataFrame of NEXRAD sites in projected geometry for shapely operations.
    
    Parameters
    ----------
    geojson_file : str
        File path to geojson file of radar sites
    crs : pyproj.CRS
        CRS of analysis

    """
    sites = gpd.read_file(geojson_file)
    sites = sites[sites["radarType"] == "NEXRAD"]
    sites['proj_geom'] = sites['geometry'].to_crs(crs)
    return sites


def filter_radar_sites(radar_sites, subbatch, r_max):
    """Obtain radar sites which have data intersecting the analysis domain box.

    Parameters
    ----------
    radar_sites : geopandas.GeoDataFrame
        Dataframe with projected geometry (label "proj_geom") to be searched for sites to
        include.
    subbatch : pandas.Series or dict-like
        Must have fields x_min, x_max, y_min, and y_max
    r_max : float
        Maximum radius of data inclusion from a radar site in units of projected geometry

    """
    radar_search_area = shapely.geometry.box(
        subbatch['x_min'] - r_max,
        subbatch['y_min'] - r_max,
        subbatch['x_max'] + r_max,
        subbatch['y_max'] + r_max
    )
    return radar_sites[[radar_search_area.contains(p) for p in radar_sites['proj_geom']]]


def create_s3_file_list(radar_site_ids, analysis_time, vol_search_interval, bucket_name='noaa-nexrad-level2'):
    """Create listing of remote NEXRAD files of interest on S3 bucket.

    Parameters
    ----------
    radar_site_ids : iterable
        Iterable of site NEXRAD site ids to query
    analysis_time : pandas.Timestamp
        Datetime of analysis
    vol_search_interval : pandas.Timedelta
        Maximum offset from analysis time for which to include files
    bucket_name : str
        Name of NEXRAD bucket on Amazon s3 (defaults to noaa-nexrad-level2)
    
    Returns
    -------
    list of str
        List of S3 file keys for Level II files of interest
    """
    s3 = boto3.client('s3')
    dates_to_search = np.unique([
        t.floor('D') for t in [
            analysis_time - vol_search_interval,
            analysis_time,
            analysis_time + vol_search_interval
        ]
    ])
    file_keys = []
    for date in dates_to_search:
        for site_id in radar_site_ids:
            all_files_on_day = [
                obj['Key'] for obj in s3.list_objects_v2(
                    Bucket=bucket_name,
                    Prefix=date.strftime(f"%Y/%m/%d/{site_id}")
                )['Contents']
            ]
            for f in all_files_on_day:
                match = l2_datetime_pattern.search(f)
                if match and (
                    analysis_time - vol_search_interval
                    <= pd.Timestamp("{Y}-{m}-{d}T{H}:{M}:{S}".format(**match.groupdict()))
                    <= analysis_time + vol_search_interval
                ):
                    file_keys.append(f)
    return file_keys


@dask.delayed
def _load_single_site(f, cache_dir, analysis_time, sweep_interval):
    try:
        radar = pyart.io.read_nexrad_archive(f)
    except:
        warnings.warn(f"Cannot read file {f}")
        return None

    sweep_time_offsets = [np.median(radar.time['data'][s:e]) for s, e in radar.iter_start_end()]
    sweep_times = cftime.num2date(sweep_time_offsets, radar.time['units'])
    valid_sweep_ids = [
        i for i, t in enumerate(sweep_times) if (
            analysis_time - sweep_interval
            <= t
            <= analysis_time + sweep_interval
        )
    ]
    if valid_sweep_ids:
        new_radar = radar.extract_sweeps(valid_sweep_ids)
        del radar
        return new_radar
    else:
        del radar
        return None


def download_nexrad_data_serial(file_keys, cache_dir, bucket_name='noaa-nexrad-level2'):
    """Download NEXRAD radar files from S3 bucket to a cache in serial fashion
    
    Parameters
    ----------
    file_keys : iterable of str
        S3 Bucket keys of Level II files of interest
    cache_dir : str
        Directory for Level II file cache
    bucket_name : str
        Bucket name on S3 for nexrad data

    Returns
    -------
    list of str
        List of filepaths to l2 files
    """
    s3 = boto3.client('s3')
    filepaths = []
    for k in file_keys:
        f = cache_dir + k.split("/")[-1]
        if not os.path.isfile(f):
            log.info(f"Downloading {f}")
            s3.download_file(bucket_name, k, f)
        filepaths.append(f)
    return filepaths


def load_nexrad_data_serial(file_keys, cache_dir, analysis_time, sweep_interval, bucket_name='noaa-nexrad-level2'):
    """Load select sweeps of NEXRAD radar files into memory from S3 bucket using a cache in a serial fashion.
    
    Parameters
    ----------
    file_keys : iterable of str
        S3 Bucket keys of Level II files of interest
    cache_dir : str
        Directory for Level II file cache
    analysis_time : pandas.Timestamp
        Datetime of analysis
    sweep_interval : pandas.Timedelta
        Maximum offset from analysis time for which to include sweeps

    Returns
    -------
    list of pyart.core.Radar
        List of subsetted Radars
    """
    s3 = boto3.client('s3')
    radars = []
    for k in file_keys:
        f = cache_dir + k.split("/")[-1]
        if not os.path.isfile(f):
            log.info(f"Downloading {f}")
            s3.download_file(bucket_name, k, f)
        radars.append(_load_single_site(f, cache_dir, analysis_time, sweep_interval))
    return [radar for radar in dask.compute(*radars) if radar is not None]
