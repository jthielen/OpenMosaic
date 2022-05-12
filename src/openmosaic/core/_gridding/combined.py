# Copyright (c) 2022 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""High-level gridding wrappers."""


def generate_mosaic_with_autoload():
    """TODO
    
    known inputs:
    - grid (GridDefinition, or something convertible to it)
    - time (something convertible to pd.Timestamp or iterable thereof)
    - fields (iterable of str, those we pull directly from pyart.core.Radar)
    - include_auxiliary_coords (bool, lat/lon option)
    - dask_client
    - verbose (bool...or also give 'html'???)
    - compute_fields (iterable of radial Calculations)

    other possible inputs:
    - s3 options (the loader itself?)
    - cache directories
    - weights, n echo, n obs fields include?
    - site to location mapping, or filtering/exclusion
    
    flow:
    - configure any default arguments not given
    - get the radar site ids to use (pre-dask)
    - build the s3 key list (pre-dask)
    - map downloads to client
    - map creation of RadarSubgrid for each LevelII downloaded to client (what does that
      do? it opens to radar file with pyart, filters to what we can use for analysis time, 
      applies any radial calculations, and sets up the subgrid coord things)
    - map creation of radar coords within each RadarSubgrid
    - map wrapper func application of map_radar_gates_to_subgrid_sum_gridrad_method
    - client submit grid.combine_subgrids() with all completed RadarSubgrid
    - client gather all datasets (across analysis time loop) and concat
    - full dataset level clean ups (post-dask)

    tbd:
    - how to clean the cache?
    """
    pass


def write_mosaics_with_autoload():
    """TODO
    
    like generate_mosaic_with_autoload, but sub times for time, and add
    - post_gridding_operations
    - output

    flow:
    - like gen..., but mapping as much of the other stuff to the dask distributed client as
      possible, then distributed writing
    """
    pass


def generate_mosaic():
    """TODO
    
    known inputs (as in generate...autoload):
    - grid
    - fields
    - compute_fields
    - dask_client
    - verbose
    - include_auxiliary_coords

    differences
    - radars (iterable of file path or pyart.core.Radar)
    - analysis_time (must be singular)

    flow:
    - configure any default arguments not given (pre-dask)
    - map creation of RadarSubgrid for each LevelII specified to client (what does that
      do? it opens to radar file with pyart, filters to what we can use for analysis time, 
      applies any radial calculations, and sets up the subgrid coord things)
    - map creation of radar coords within each RadarSubgrid
    - map wrapper func application of map_radar_gates_to_subgrid_sum_gridrad_method
    - client submit grid.combine_subgrids() with all completed RadarSubgrid and gather it
    - full dataset level clean ups (post-dask)

    """