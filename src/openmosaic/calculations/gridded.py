# Copyright (c) 2021 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Radar field calculations to be applied post-gridding."""

import xarray as xr


class BaseGridCalculation:
    # TODO: Docstring
    def __init__(self):
        pass

    def check_data(self, data):
        if (
            not isinstance(data, xr.Dataset) or "altitude" not in data.coords
            or "altitude" not in data.dims
        ):
            raise ValueError(
                "data must be an xarray Dataset output from a standard OpenMosaic gridding "
                "function"
            )


class LayerMaximum(BaseGridCalculation):
    # TODO: Docstring

    def __init__(self, fields, layers=None, layer_labels=None, drop=False):
        # TODO: docstring

        # Take fields and drop as-is
        self.fields = fields
        self.drop = drop
        
        # Validate layers
        if (
            layers is not None and (
                not isinstance(layers, list) or
                any(not isinstance(layer, tuple) or len(layer) != 2 for layer in layers)
            )
        ):
            raise ValueError(
                "layers argument must be None (for full vertical maximum) or a list of "
                "2-tuples"
            )
        elif layers is None:
            self.layers = [(None, None)]
        else:
            self.layers = layers

        # Validate layer_labels
        if self.layers[0][0] is None:
            if layer_labels is None:
                self.layer_labels = ["vertical_maximum"]
            elif isinstance(layer_labels, str):
                self.layer_labels = [layer_labels]
            else:
                raise ValueError(
                    "layer_labels, if specified, must be a string when computing only the "
                    "vertical maximum"
                )
        else:
            if layer_labels is None or len(layer_labels) != len(self.layers):
                raise ValueError(
                    "layer_labels must be specified as a collection of strings matching that "
                    "of layers when layers are specified"
                )
            else:
                self.layer_labels = tuple(str(label) for label in layer_labels)

    def __calc__(self, data):
        # TODO: docstring
        self.check_data(data)

        for (layer_min, layer_max), label in zip(self.layers, self.layer_labels):
            for field in self.fields:
                data[f"{label}_{field}"] = data[field].sel(
                    altitude=slice(layer_min, layer_max)
                ).max('altitude')

        if self.drop:
            data = data.drop_vars(self.fields)            

        return data

