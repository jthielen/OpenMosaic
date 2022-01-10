# Copyright (c) 2021 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""...TODO..."""

import pandas as pd
import requests
import xarray as xr

from .core import DatasetLoader


def _create_http_auth_session(username, password):
    # TODO docstring
    session = requests.Session()
    session.auth = (username, password)
    return session


class RDADatasetLoader(DatasetLoader):
    # TODO: docstring

    def __init__(self, username=None, password=None, *args, **kwargs):
        # TODO: docstring
        if username is None or password is None:
            raise ValueError("RDA requires a username and password for authentication.")
        
        self.session = _create_http_auth_session(username, password)

        super().__init__(*args, **kwargs)

    def _get_source_store(self, path):
        # TODO docstring
        return xr.backends.PydapDataStore.open(path, session=self.session)


class ERA5Loader(RDADatasetLoader):
    # TODO docstring

    base_format = (
        "https://rda.ucar.edu/thredds/dodsC/files/g/ds633.0/e5.oper.an.pl/%Y%m/"
        "e5.oper.an.pl.128_130_{field_abbr}.ll025sc.%Y%m%d00_%Y%m%d23.nc"
    )
    field_abbreviations = {
        "geopotential": "z",
        "air_temperature": "t",
        "eastward_wind": "u",
        "northward_wind": "v"
    }
    timestep = pd.Timedelta(1, 'hr')
    dependent_fields = {
        "altitude_at_freezing_level": ("geopotential", "air_temperature"),
        "altitude_at_-20degC_surface": ("geopotential", "air_temperature"),
        "eastward_wind_on_altitude_levels": ("geopotential", "u"),
        "northward_wind_on_altitude_levels": ("geopotential", "v"),
    }

    def __init__(self, *args, **kargs):
        # TODO docstring
        super().__init__(*args, **kargs)

    def _get_url(self, field, time):
        if field not in self.field_abbreviations:
            return ValueError(f"Unrecognized field {field}")
        else:
            return time.strftime(
                self.base_format.format(field_abbr=self.field_abbreviations[field])
            )

    def _get_source_data(self, requested_field, time):
        if (
            requested_field not in self.field_abbreviations
            and requested_field not in self.dependent_fields
        ):
            raise ValueError(f"Unrecognized field {requested_field}")

        # todo finish implementation
        pass

    def load_data_like(self, requested_field, data_like):
        # todo docstring
        # todo implement
        pass

    # TODO will likely need to add: isobaric to altitude interpolation
        