# Copyright (c) 2020 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Generic shared utils."""

import os
import re

import cftime
import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import pyart
import shapely
from tqdm import tqdm


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


def create_s3_file_list(nexrad_bucket, radar_site_ids, analysis_time, vol_search_interval):
    """Create listing of remote NEXRAD files of interest on S3 bucket.

    Parameters
    ----------
    nexrad_bucket : boto3.resources.factory.s3.Bucket
        NEXRAD Level II S3 Bucket object
    radar_site_ids : iterable
        Iterable of site NEXRAD site ids to query
    analysis_time : pandas.Timestamp
        Datetime of analysis
    vol_search_interval : pandas.Timedelta
        Maximum offset from analysis time for which to include files
    
    Returns
    -------
    list of str
        List of S3 file keys for Level II files of interest
    """
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
                obj.key for obj in nexrad_bucket.objects.filter(
                    Prefix=date.strftime(f"%Y/%m/%d/{site_id}")
                )
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
    radar = pyart.io.read_nexrad_archive(f)
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
        return radar.extract_sweeps(valid_sweep_ids)


def load_nexrad_data(file_keys, nexrad_bucket, cache_dir, analysis_time, sweep_interval):
    """Load select sweeps of NEXRAD radar files into memory from S3 bucket using a cache.
    
    Parameters
    ----------
    file_keys : iterable of str
        S3 Bucket keys of Level II files of interest
    nexrad_bucket : boto3.resources.factory.s3.Bucket
        NEXRAD Level II S3 Bucket object
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
    radars = []
    for k in file_keys:
        f = cache_dir + k.split("/")[-1]
        if not os.path.isfile(f):
            warnings.warn(f"Downloading {f}")
            nexrad_bucket.download_file(k, f)
        radars.append(_load_single_site(f, cache_dir, analysis_time, sweep_interval))
    return [radar for radar in dask.compute(*radars) if radar is not None]
