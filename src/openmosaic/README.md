# Development Notes

## September 2022 Development Sprint

### High-Level API Planning

#### Brainstorming

##### Usage Templates

**Rankine Vortex Test**

- Create a rankine vortex
- Compute rVd, MRMS-AziShear, and GridRad-AziShear in radar coords
- Grid to an automatically determined 2D grid
- Compute DVAD-LLSD-Quasi-Vorticity in gridded coords
- Have a nice output dataset comparing three methods for AziShear-like field in gridded space

```python
from openmosaic import generate_mosaic
from openmosaic.calculations import DvadLlsd, Llsd, SmoothedDerivatives
from openmosaic.grid import GridDefinition
from openmosaic.testing import rankine_simulate

ds_radar = rankine_simulate(
    7.5e2, 30, 3.0e4, 3.6e4, 2.0e4, 2.6e4, delta_cartesian=1e2
)
target_grid = GridDefinition.simplified(dx=3.0e2, dy=3.0e2)
ds_gridded = generate_mosaic(
    ds_radar,
    grid=target_grid,
    time=None,
    fields=None,
    radar_compute_fields=[
        DvadLlsd(prefix='dvad_llsd_').compute_rvd,
        Llsd(
            azimuthal_shear=True,
            radial_divergence=False,
            prefix='llsd_'
        ),
        SmoothedDerivatives(
            azimuthal_shear=True,
            radial_divergence=False,
            prefix='gridrad_'
        )
    ],
    subgrid_compute_fields=[DvadLlsd(prefix='dvad_llsd_', drop=True).vorticity_from_rvd],
    grid_compute_fields=None
)
```

**CONUS NEXRAD Snapshot**

- Set a time
- Use a standard grid
- Just have composite reflectivity

```python
from openmosaic import generate_mosaic
from openmosaic.calculations import LayerMaximum
from openmosaic.grid.standard import Conus2km
from openmosaic.patterns import NexradS3

time = '2020-08-10 16:00Z'
ds_gridded = generate_mosaic(
    NexradS3,
    grid=Conus2km,
    time=time,
    fields=['reflectivity'],
    grid_compute_fields=[LayerMaximum(fields=['reflectivity'], prefix='composite_', drop=True)]
)
```

**Replicate GridRad for some interval of time**

- Specify times
- Prepare a recipe-ready pattern for given times and grid from a generic pattern (this should
  solve what S3 files should be requested and organize them for subgrid gridding and mosaicing)
- Define the recipe from prepared pattern and mosaic control options (following kwargs of
  generate_mosaic previously)
- Run the recipe using desired executor

```python
from openmosaic.calculations.diagnostics import GridRadDiagnostics
from openmosaic.grid.standard import GridRadv4p2
from openmosaic.patterns import NexradS3
from openmosaic.recipes import ZarrMosaicRecipe

times = pd.date_range('2022-06-01', '2022-07-01', freq='H')
prepared_pattern = NexradS3.prepare_from_grid(time=times, grid=GridRadv4p2)
recipe = ZarrMosaicRecipe(
    prepared_pattern,
    mosaic_options={
        'grid': GridRadv4p2,
        'fields': ['reflectivity', 'spectrum_width'],
        'diagnostics': [GridRadDiagnostics(nradobs=True, nradecho=True)]
    },
    # TODO: specify zarr output compression in place of compression-by-gathering, which
    # doesn't stack, etc.
)

recipe.to_dask()
```

#### Process Outline

##### General steps

(for each time)

1) Load radar data (and compute needed metadata)
2) Perform radar-space calculations (if any; e.g. velocity dealiasing, LLSD AziShear)
3) Grid volume(s) from single radar to intermediate subgrid
4) Perform subgrid/Cartesian-space calculations (if any; e.g., DVAD-LLSD Vorticity)
5) Reduce subgrid to parent grid resolution (if refined)
6) Merge subgrids across space
7) Perform full grid calculations (e.g., echo tops)
8) Save result

##### Simplfied steps

**Single Time**

(same as general, but without time-loop)

**Single Radar**

1) Load radar data (and compute needed metadata)
2) Perform radar-space calculations (if any; e.g. velocity dealiasing, LLSD AziShear)
3) Grid volume(s) from single radar to grid
4) Perform grid calculations (if any; e.g., DVAD-LLSD Vorticity, echo tops, coarsening)
5) Save result

##### Options/Control of Process

- Data Input
    - Options:
        - xradar object
        - callable returning an xradar object
        - some form of datatime & NEXRAD location specification
- Mosaic Specification
    - Output grid definition
    - Intermediate subgrid control
- Calculations

#### API Outline

- `calculations`
- `core`
- `executors`
- `geo`
- `grid`
- `patterns`
- `recipes`
- `testing`
- `storage`

### Misc Notes

- `testing` has a MetPy dependency now


### Early-On Reflection

So...summer 2022 plans did *not* go well at all given personal health issues and all that. Here's to hoping that September can be when things turn around.

Not really sure how to get back into all this, except just scattershot and plow through the list of things that need to be done for this to be a useable package, with four main targets in mind:

- Make the demonstrations/executable notebook preprint of my DVAD-LLSD work really slick and seamless
- Provide easy-to-use tooling for Gallus's group's project
- Finally deliver on goals from sponsored funding from Haberlie group
- Have documentation sufficient for a JOSS submission for OpenMosaic

Since it's foremost on my mind, I kind of want to start out by folding back in the components of my DVAD-LLSD stuff.

**Somewhat Later Goals**:

- Revisit `core/_gridding/_PENDING_REMOVAL*` files and verify that all functionality used there has been re-implemented via Numba elsewhere in `core/_gridding/`
- Build a simple OSSE test workflow ("true" field, detect with simple fake radar, gridding and calcs, compare with original)
- Implement unit tests of all the stuff that's "local" (no remote data loading, no workflow integration tests yet)

**More Later Goals**:

- Revisit the "generate full CONUS NEXRAD mosaic for given time" use case and verify that all the "plumbing" works for it
- Investigate test coverage and go through plans to have it all fully covered
- Make sure public API documentation is complete
- Document "rough edges" features (capabilities designed for, but not fully worked out in initial release version)
    - Also roadmap for incomplete or unimplemented features
- Document basic use cases supported by initial release
- Actually release it on conda-forge, etc.

--------------------------------------------------------------------------

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
- More advanced dealias algo support (including fetching of analysis profiles or obs soundings or linking to previous volumes)

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

### Miscellaneous In-Progress Notes

#### Up Next

**Immediate**

- Refactor the DaskPipelineExecutor to use futures interface and allow nested Pipelines
- Revisit the recipe stack (top down this time) to make the class structure work out best

**Later, but still in mind**

- Get rid of all the `_PENDING_REMOVAL_` files (i.e., verify that all such code is adequately incorporated elsewhere)
- Brainstorm how your calculation framework interfaces with recipe framework
- Implement an RDA-friendly fsspec interface

#### Code formatting

- using blue, with line length modified to 95
- not yet run, will wait for old code to be cleared out first
