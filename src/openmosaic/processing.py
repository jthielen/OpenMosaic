# Copyright (c) 2020 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Batch creation and handling and other processing related utils."""

import geopandas
import numpy as np
import pandas as pd


def hello_world():
    """Test that tests and project configuration work. TODO: Remove"""
    return np.mean([41, 43])


def split_by_time_gap(df, datetime_col, time_delta):
    """Partition a DataFrame into datetime groups separated by a time gap.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with a datetime column to be partitioned
    datetime_col : str
        Column label for the datetime column to use
    time_delta : numpy.timedelta64
        Minimum spacing between datetime groups. Any rows of ``df`` within ``time_delta`` of
        each other will be within the same group.

    Returns
    -------
    list
        List of individual dataframes representing the split groups

    """
    return np.split(
        df.sort_values(datetime_col),
        np.argwhere((df[datetime_col].sort_values().diff() > time_delta).to_numpy()).flatten(),
    )


def prepare_snapshots(
    df,
    datetime_col="datetime",
    analysis_freq="H",
    count_before=0,
    count_after=0,
    id_column="unique_id",
    lon_column="lon",
    lat_column="lat",
):
    """Expand individual location/time rows across time based on analysis frequency.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with rows containing entries to be expanded across time
    datetime_col : str, optional
        Column of dataframe with datetime. Defaults to 'datetime'.
    analysis_freq : str, optional
        Frequency on which analyses are conducted. All datetimes are associated with the
        nearest time on regular intervals according to this frequency, and before/after
        snapshots are separated by this frequency. Uses Pandas's time frequency parlance.
        Defaults to 'H' (hourly).
    count_before : int, optional
        Number of additional snapshots before rounded time at the location. Defaults to 0.
    count_after : int, optional
        Number of additional snapshots after rounded time at the location. Defaults to 0.
    id_column : str, optional
        Column of preserved id label (for cross-referencing output with original input
        dataframe). Defaults to 'unique_id'.
    lon_column : str, optional
        Column label of longitude. Defaults to 'lon'.
    lat_column : str, optional
        Column label of latitude. Defaults to 'lat'.

    Returns
    -------
    geopandas.DataFrame
        A geopandas dataframe with point geometry column added based on lons/lats, with rows
        corresponding to the temporally rounded and expanded data.

    """
    # Build time range
    timedeltas = pd.timedelta_range(
        start=0, periods=1 + count_before + count_after, freq=analysis_freq
    )
    timedeltas -= timedeltas[count_before]

    # Collect snapshots
    snapshots = []
    for _, row in df.iterrows():
        nearest_time = row[datetime_col].round(analysis_freq)
        snapshots += [
            (row[id_column], row[datetime_col], row[lon_column], row[lat_column], time)
            for time in nearest_time + timedeltas
        ]

    # Assemble into new dataframe
    df = pd.DataFrame.from_records(
        snapshots, columns=[id_column, datetime_col, lon_column, lat_column, "analysis_time"]
    )
    return geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df["lon"], df["lat"], crs="EPSG:4326")
    )


def get_subdomain_splits(geometry, buffer):
    """Split geometry collection-of-points into rectangular subdomains.

    Parameters
    ----------
    geometry : geopandas.GeoSeries
        GeoSeries containing the points to be contained.
    buffer : pandas.Series or float
        One-half the maximum x/y distance separating points within the same subdomain. Works
        by acting as a radius, which then defines a box, and non-overlaping subsets of the
        unary union of these boxes are the subdomain regions.

    Returns
    -------
    pandas.Series
        Integer labels cooresponding to the subdomains of the input points once split.

    """
    subsets = geometry.buffer(buffer).envelope.unary_union
    regions = (
        geopandas.GeoDataFrame(geometry=[subsets])
        .explode()
        .reset_index(drop=True)
        .geometry.envelope
    )
    return pd.Series([regions.contains(p).idxmax() for p in geometry], geometry.index), regions
