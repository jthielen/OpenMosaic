# Copyright (c) 2020 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Test calculation functionality of OpenMosaic."""

from openmosaic.routines import hello_world


def test_routines_filler():
    """Test hello world function."""
    assert hello_world() == 42
