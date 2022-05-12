# Copyright (c) 2021 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Loading functionality for data on AWS S3."""

import logging
import os
import re
import warnings

import boto3
import cftime
import dask
import numpy as np
import pandas as pd
import pyart

from .core import LevelIILoader

class S3LevelIILoader(LevelIILoader):
    # TODO docstring
    def __init__(self, *args, **kwargs):
        # TODO docstring
        # TODO allow bucket/boto3 control, but enable smart defaults
        super().__init__(*args, **kwargs)

    """
    TODO
    
    - Define the valid files types to check for (and how they map to any needed preprocessing)
    - Scan for available files for site given time range (a higher level utility will use this
      to determine if there are enough sites present to have a meaningful mosaic)
    - Download a file from s3 to local cache (unzipping if gzipped)
    - Open source LII files to pyart radars (can hopefully reuse from super, possibly munging
      path)
    - Any extra downloaded file cache management (i.e., can delete the level II file once the
      processed cf-radial is saved.)
    - ...otherwise everything else should follow from the base loader...
    """


##### COPY FROM OLD utils.py #####

# Copyright (c) 2020 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
log = logging.getLogger(__name__)
l2_datetime_pattern = re.compile(
    r"(?:K[A-Z]{3})(?P<Y>[0-9]{4})(?P<m>[0-9]{2})(?P<d>[0-9]{2})_(?P<H>[0-9]{2})"
    r"(?P<M>[0-9]{2})(?P<S>[0-9]{2})"
)


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
            s3_search = s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix=date.strftime(f"%Y/%m/%d/{site_id}")
            )
            if 'Contents' not in s3_search:
                warnings.warn(date.strftime(f"No files found for {site_id} on %Y-%m-%d"))
                continue
            all_files_on_day = [
                obj['Key'] for obj in s3_search['Contents']
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
