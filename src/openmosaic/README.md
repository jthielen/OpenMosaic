# Development Notes

## May 2022 Development Sprint

### New "Code Plan"

Since https://github.com/jthielen/OpenMosaic/discussions/11, I've reconsidered the overall package plan based on needs and inspiration from elsewhere in the ecosystem. Here is getting a summary of my thoughts down for record (and later self-reminder):

#### Pangeo Forge Inspiration

From a high-level perspective, much of what OpenMosaic seeks to accomplish is a detailed & complicated version of pangeo-forge: retrive remote data, process the data, and output in regular/aligned form, all in a highly parallelized and scalable way. So, I took inspiration from the recipe/executor structure of pangeo-forge to guide the core workflow of using OpenMosaic, hence the analogous modules and contents:

- `executors` (managing how a recipe is run)
    - `DaskPipelineExecutor`
    - `FunctionPipelineExecutor`
    - `GeneratorPipelineExecutor`
- `recipes`
    - `BaseMosaicRecipe` (everything in common)
    - `XarrayMosaicRecipe` (output to single, in-memory xarray.Dataset)
    - `ZarrMosaicRecipe` (append to a Zarr store)
    - `MultifileMosaicRecipe` (for each time, dump a file according to a Path pattern)
- `patterns` (interfacing with target file resources)
- `storage` (generic interfaces with file inputs and outputs, probably through fsspec)

One of the main hopes in following Pangeo Forge's API/design pattern is that improvements upstream to workflow handling can be easily brought down to OpenMosaic at a later time (such as setting up a real-time streaming workflow with Prefect, or authentication/secret handling with common data sources like NCAR RDA). A secondary one is code reuse (which could save dev efforts).

This all being said, I still plan to provide a top-level `generate_mosaic` function (which, under the hood will be a `XarrayMosaicRecipe` conditionally using `DaskPipelineExecutor` or `FunctionPipelineExecutor`) so that users can just pass their mosaic specifications and easily get a single mosaic out. I think it will be just this one, though, as the `generate_mosaic_with_autoload` version can really just be built in (if files are unspecified), and the `write_mosiacs_with_autoload` is better suited towards direct use of the `ZarrMosaicRecipe` or `MultifileMosaicRecipe`.

#### Calculations

A big motivator for this development sprint is my summer research plans at CSU, involving the development of a synthesis of DVAD and LLSD approaches for creating approximations to vorticity and divergence from Doppler velocity (and comparing back to GridRad and MRMS approaches for AziShear and RadDiv). And so, while Echo Top Height, SHI/MESH, and general layer aggregation will still be included in the initial release, there's going to be a big focus on Doppler velocity calculations. Luckily, these together should help create a calculation framework where calculations can suitably operate on
- origin (or rotated/standardized) radar polar grid
- intermediate rectangular/quasi-Cartesian grid
- full mosaic

I'm all-in on using numba (over Cython or otherwise) to provide optimized routines, and given the workflow style approach, I think it would be best to structure this with

- `_routines/*.py`, for just the raw functions without grid/radar object handling or option specification
- the main files, with "function generators" (i.e., `operation(options=...)(actual_data)`) built either as classes or as decorated functions creating a partial if `data` is omitted

All in all, this will all be encapsulated in the `calculations` first-level module.

#### Testing

A neat side effect (which will also be useful in my own research) of recent coursework is creating a Rankine Vortex/"Burst" simulator for OSSEs. Mind as well include that here for the test suite and for others to use. API for this is likely to change, but will be housed in the `testing` first-level module.

#### Other Aspects of Overall API

Other first-level modules not previously

- `core`
    - any shared utilities
    - the actual GridRad-style gridding routine(s)
- `geo`
    - small module just for geographic/nexrad site helpers
- `grid`
    - `GridDefinition` (encapsulating how a grid is specified, which *new*, will be including a `parent` reference, since I'll be having need to support an intermediate rectangular/quasi-Cartesian grid for each radar volume given some kinds of calculations, perhaps with refinement of grid spacing)
    - All the stuff for transforming from the radar's Azimuthal Equidistant x/y coords to the mosaic's projection x/y coords (true/base pyproj Transformer and shortcut ElasticNet approximator)

### Recent Ideas, but Defering Action

- Support for gridding methodologies other than GridRad-style nearest-neighbor (such as Cressman/Barnes, variational, or a port of whatever LROSE uses)
- Getting your azishear/raddiv implementations pushed upstream to PyART or in their own package (as they might be useful outside of the context of mosaicing?)
- Directly engaging the pangeo-forge folks to say "here's what I'm trying to do, like a modification on pangeo-forge's approach, can we discuss refactoring pangeo-forge-recipes so it can be extended by downstream users making similar data pipelines with more esoteric needs"

### Target Goals/Timeline

Phase 1) May Development Sprint

*Pendulum towards code development focus*

- Have all code drafted
- Have unit tests implemented of all the lowest level stuff

Phase 2) Integrating Into Research

*Pendulum towards research focus*

- Throughly investigate Doppler velocity derivative calculations, resulting in
    - any bugs fixed
    - extra (integration-level) tests with that part of the 
    - thorough docstrings and related write-ups explaining all relevant methodologies
    - start of examples for documentation showing rankine simulation (and other OSSE) and real-world use
- Most of a manuscript prepared, except for the stuff needed a released version of OpenMosaic (data avaiability, accompanying code repo, etc.)

Phase 3) Preparing for initial release

*Pendulum towards code development focus*

- Review test coverage and CI, and make necessary improvements
- Create docs (and deploy using GitHub actions/pages a la MetPy)
- Create tooling for release pipeline on conda forge and pypi
- Actually release it!

Phase 4) Publication time

*Pendulum in the middle*

- Finalize the DVAD-OSSE manuscript and submit to JTECH
- Improve OpenMosaic docs based on collaborator (and hopefully user!) feedback
- Follow the pre-submission process for JOSS (particularly https://joss.readthedocs.io/en/latest/submitting.html#co-publication-of-science-methods-and-software)
- Submit OpenMosaic to JOSS

Phase 5) Wait

...wait...

...then, boom, it's mid-August and another academic year begins!
