# Copyright (c) 2021 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Radial and cartesian grid radar field calculations.

TODO: this will be reorganized significantly.

Define here:

- RadarCalculation
    - base class for calculations occuring in azimuth-range space
- GridCalculation
    - base class for calculations occuring in x-y space
    - generic for both refined/sub-grid and parent-grid
- DvadLlsd
    - CalculationFactory for your own research's DVAD-LLSD methods
    - `.compute_rvd` to give a RadarCalculation for rVd
    - `.vorticity_from_rvd` to give a GridCalculation for quasi-vorticity from gridded rVd
    - others tbd (probably enumerate full combo of vorticity/divergence, pre/post gridding)
    - __call__ tbd?
- Llsd
    - CalculationFactory for RadarCalculation for MRMS-style LLSD methods for azishear/raddiv
    - make generic for azimuthal and radial derivatives of other moments?
    - specify options in __call__
    - direct methods tbd?
- SmoothedDerivatives
    - CalculationFactory for RadarCalculation for GridRad-style smoothing + finite difference
      for azishear/raddiv
    - make generic for azimuthal and radial derivatives of other moments?
    - specify options in __call__
    - direct methods tbd?
- LayerMaximum
    - CalculationFactory for GridCalculation for reducing 3D to 2D using maximum
    - specify options in __call__
    - direct methods (like oft-used composite_reflectivity) tbd?
"""

# from .gridded import PyARTFilter, SpecificDifferentialPhase, VelocityDerivatives
# from .radial import EchoTopHeight, LayerMaximum, MESH, SHI
