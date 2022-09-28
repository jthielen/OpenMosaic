# OpenMosaic: Open-source and extensible radar mosaic creation in Python

OpenMosaic is a toolkit for creating radar mosaics (in the fashion of [MRMS](https://vlab.noaa.gov/web/wdtd/mrms-products-guide/), [MYRORRS](https://osf.io/9gzp2/), and [GridRad](http://gridrad.org/)) that leverages many packages in [the Pangeo ecosystem](https://pangeo.io/). While OpenMosaic [was initially built](https://www.youtube.com/watch?v=OQlnL_h8PYM) for NEXRAD mosaics in particular, the general features can be used for any collection of stationary radars.

Development is ongoing, and feedback and contributions are welcome! Contact [@jthielen](https://github.com/jthielen) with questions.

## Core Features

- Radar data gridding (from azimuth-range to x-y space)
- Calculations (in radar and/or grid space) such as
    - Azimuthal Shear (with several choices of algorithms)
    - Echo Tops
    - and many more
- Dask-based parallelization of radar gridding tasks
- Metadata-rich outputs
- Two main APIs
    - A simplified `generate_mosaic` function for creating a single mosaic
    - A [Pangeo Forge](https://pangeo-forge.readthedocs.io/en/latest/) inspired recipe framework for large mosaic creation workflows

## Relevant Packages in the Ecosystem

OpenMosaic utilizes and/or integrates with packages such as:

- [xarray](https://xarray.dev/), providing the data model for mosaic output
- [xradar](https://docs.openradarscience.org/projects/xradar/) and [datatree](https://github.com/xarray-contrib/datatree), providing the data model for radar data input (either user-provided or internal)
- [Py-ART](https://arm-doe.github.io/pyart/) for radar-based calculations
    - Also provides an alternative data model for radar data input while xradar is under development
- [fsspec](https://filesystem-spec.readthedocs.io/), for cloud-aware file access (particularly automatic downloading of NEXRAD Level II data from AWS S3)
- [geopandas](https://geopandas.org/) and [pyproj](https://pyproj4.github.io/pyproj/) for geospatial utilities
- [MetPy](https://unidata.github.io/MetPy/) for features used in examples and testing

The widely-used [numba](https://numba.pydata.org/) and [scikit-learn](https://scikit-learn.org/) packages are also leveraged for internal performance optimizations.

Also, while it is not used or integrated directly, the processing framework used by OpenMosaic is built off of that of [Pangeo Forge](https://pangeo-forge.readthedocs.io/en/latest/). Direct integration with Pangeo Forge may be possible in the future.

## License Information

Copyright 2021-2022, OpenMosaic Developers.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Additional Information

If you've seen OpenMosaic prior to late 2022, its initial scope included functionality for "ML-ready feature extraction" from radar mosaics. This functionality will be released elsewhere at a future point in time.
