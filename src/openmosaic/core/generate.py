# Copyright (c) 2022 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Shared routines for generation of a simple mosaic.

TODO: actually implement, rather than spec-ing out
"""


def generate_mosaic(
    *inputs,
    grid,
    time=None,
    fields=None,
    radar_compute_fields=None,
    subgrid_compute_fields=None,
    grid_compute_fields=None,
    subgrid_options=None
):
    """Generate a single radar mosaic.

    Parameters
    ----------
    inputs : datatree.DataTree or openmosaic.patterns.InputPattern
        Input data from which to create the mosaic, either given directly as an xradar
        DataTree (or collection thereof) or InputPattern template for loading
    grid : openmosaic.grid.GridDefinition
        Target grid for the mosaic. Can be fully defined or left partially parametric to
        fit to inputs.
    time : pandas.Timestamp (or input thereof)
        Time for which the mosaic is valid. If not provided, will mosaic without respect to
        time variation. Must be provided if an InputPattern is used, optional if data are
        directly given.
    fields : iterable of str, optional
        List of field labels to grid directly from input data. If using subgrid_compute_fields
        or grid_compute_fields, inputs to said calculations must be given here.
    radar_compute_fields : iterable of openmosaic.calculations.RadarCalculation
        List of calculations to apply on radar data prior to gridding. Applied in order, so
        calculations that provide fields for other calculations must come first. Inputs need
        not be listed in fields.
    subgrid_compute_fields : iterable of openmosaic.calculations.GridCalculation
        List of calculations to apply on gridded data immediately after gridding, but prior to
        mosaicing between radars. Allows "Cartesian" calculations on a finer grid than the
        output, if using a refined subgrid.
    grid_compute_fields : iterable of openmosaic.calculations.GridCalculation
        List of calculations to apply following mosaicing all the sub-grids from different
        radars. 3D to 2D reduction operations best occur here.
    subgrid_options : dict
        Options for construction of per-radar subgrids from the given grid. TODO: determine
        what these are. Current ideas: refine_horizontal (integer multiple for horizontal
        refinement), ...

    Notes
    -----
    TODO: consider adding diagnostics (so as to copy GridRad's nradobs, nradecho)
    """
    ...
