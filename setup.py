# Copyright (c) 2020 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Setup script for installing OpenMosaic."""

from setuptools import Extension, setup

setup(
    use_scm_version={'version_scheme': 'post-release'},
    ext_modules=[
        Extension(
            'openmosaic.gridding._gate_to_grid_map',
            sources=[
                'src/openmosaic/gridding/_gate_to_grid_map.pyx'
            ],
            libraries=['m']
        ),
        Extension(
            'openmosaic.gridding._map_gates_to_subgrid',
            sources=[
                'src/openmosaic/gridding/_map_gates_to_subgrid.pyx'
            ],
            libraries=['m']
        ),
        Extension(
            'openmosaic._c_routines',
            sources=[
                'src/openmosaic/_c_routines.pyx'
            ],
            libraries=['m']
        )
    ]
)

