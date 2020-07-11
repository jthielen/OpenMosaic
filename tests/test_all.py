# Copyright (c) 2020 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Test all functionality of OpenMosaic."""

from openmosaic import hello_world


def test_hello_world():
    """Test hello world function."""
    assert hello_world() == 42
