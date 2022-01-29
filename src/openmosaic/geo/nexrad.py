# Copyright (c) 2022 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Geographic helpers for NEXRAD sites."""

import geopandas as gpd
import shapely


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
    return sites.set_index('siteID', drop=False)


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