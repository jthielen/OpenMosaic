# Copyright (c) 2021 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""...TODO..."""

import xarray as xr


class DatasetLoader:
    # todo docstring
    def __init__(self) -> None:
        # todo docstring
        # todo add args: path=None, cache=False, regrid_base_fields=True, dataset_id=None, mfdataset_kwargs
        # todo implement
        # (note: if cache is True, choose a default...if string, make a Path object)
        pass

    def get_like(self, requested_field, data_like):
        # todo docstring
        if self.check_cache(requested_field, data_like):
            return self.load_from_cache(requested_field, data_like)
        else:
            return self.load_data_like(requested_field, data_like)

    def check_cache(self, requested_field, data_like):
        # todo docstring
        if not self.cache:
            return False
        else:
            return all(
                (self.cache / f"{self.dataset_id}_{requested_field}_{time}.nc").exists()
                for time in data_like['time'].dt.strftime("%Y%m%d_%H%M%SZ")
            )

    def load_from_cache(self, requested_field, data_like):
        # TODO docstring
        # TODO any checks that it matches data_like?
        return xr.open_mfdataset(
            (
                self.cache / f"{self.dataset_id}_{requested_field}_{time}.nc"
                for time in data_like['time'].dt.strftime("%Y%m%d_%H%M%SZ")
            ),
            concat_dim="time"
        )

    def load_data_like(self, requested_field, data_like):
        # todo docstring
        if self.path is None:
            raise ValueError("Base DatasetLoader requires a path from which to load dataset(s)")

        try:
            ds_combined = xr.open_mfdataset(self.path, concat_dim="time", **self.mfdataset_kwargs)
            # TODO REGRID AND CACHE!!
            return ds_combined[requested_field].sel(time=data_like["time"])
        except:
            raise ValueError(
                f"{requested_field} must be present in source dataset(s), and time "
                "coordinates of source data and target data must correspond."
            )

    # TODO add: cache helpers (e.g., allow expiry to save space), reusable horizontal regrid


class LevelIILoader:
    # todo docstring
    def __init__(self) -> None:
        # todo docstring
        # todo add args: path=None, cache=False, dataset_id=None
        # todo implement
        # (note: if cache is True, choose a default...if string, make a Path object)
        pass

    """
    TODO:

    - Load from local into pyart radar
    - Calculate target grid x/y coords and altitude z (and so encompass that part of what was
      the "gridding" module)
    - Save local (with grid coords) as CF/Radial nc
    - Load cached CF/Radial nc
    - Time range handling (and record caching)
    """
