# Copyright (c) 2020 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Test gridding functionality of OpenMosaic."""

from openmosaic.gridding import hello_world


def test_gridding_filler():
    """Test hello world function."""
    assert hello_world() == 42
