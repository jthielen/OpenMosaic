# Copyright (c) 2022 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Primary workflow recipies for structing the process of mosaic creation.

Inspired by pangeo-forge.

TODO: recipes to create
    - CommonMosaicRecipe (everything in common)
    - XarrayMosaicRecipe (output to single, in-memory xarray.Dataset)
    - ZarrMosaicRecipe (append to a Zarr store)
    - MultifileMosaicRecipe (for each time, dump a file according to a Path pattern)
"""
