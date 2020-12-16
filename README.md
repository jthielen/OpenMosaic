# OpenMosaic: Open-source and extensible NEXRAD mosaic creation in Python

Multi-radar analyses from the NEXRAD WSR-88D network such as MRMS, MYRORRS, NOWrad, and GridRad are widely used products in the atmospheric sciences research community, especially in severe weather research. However, these existing products have particular spatial resolutions, domain coverages, analysis timesteps, product types, historical availabilities, and archive accessibilities that may not meet the needs of a particular given research problem. Additionally, most of these operational multi-radar mosaic products rely upon closed-source tools, making replication of these products for custom configurations difficult.

By leveraging and extending the IO and gridding utilities of the open-source Python ARM Radar Toolkit (Py-ART), OpenMosaic establishes an extensible framework for the creation of NEXRAD mosaic products from freely-available Level II radar data. The core of OpenMosaic is its implementation of the 4D GridRad mosaicing algorithm that leverages dask for parallelization and scalability. Also included in OpenMosaic are utilities for scanning cloud data stores of Level II data, defining georeferenced output targets from storm report or other point data, batching processing jobs, calculating derived products such as azimuthal shear and echo classification, outputting CF-compliant datasets, and applying storm object identification and feature data extraction methods based on scikit-image and hagelslag.

Development is ongoing, and feedback and contributions are welcome. Contact [Jon Thielen](https://github.com/jthielen) with questions.

## License Information

Copyright 2020 OpenMosaic Contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
