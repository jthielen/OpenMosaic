# Copyright (c) 2022 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Base class(es) for defining an OpenMosaic recipe, modeled after pangeo-forge.

This file contains code originating from pangeo-forge-recipes, (c) 2022 pangeo-forge-recipes
developers. Used with modification under the terms of the Apache-2.0 license (see LICENSE for
full text of license). Modifications (to-date) entail docstring/comment modifications, changes
to imports (based on differing APIs), and deletion of unsupported methods. Otherwise, all
functional code that is present is taken as-is from pangeo-forge-recipes.

TODO:
- update docstrings to NumPy conventions as with remainder of OpenMosaic
- consider splitting the pangeo-forge bases from BaseMosaicRecipe stuff
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field, replace
from typing import Any, Callable, ClassVar, Hashable

from ..executors.base import Pipeline
from ..patterns import FilePattern, prune_pattern
from ..core.serialization import dataclass_sha256
from ..storage import StorageConfig, temporary_storage_config

##########################################
# Base Implementations from pangeo-forge #
##########################################

@dataclass
class BaseRecipe(ABC):
    """Base class for recipes.

    Reused from pangeo-forge-recipes, with deletion of `to_prefect` and `to_beam` methods,
    which are not supported by OpenMosaic at this time.
    """
    _compiler: ClassVar[RecipeCompiler]
    _hash_exclude_ = ["storage_config"]

    def to_function(self):
        from ..executors.python import FunctionPipelineExecutor

        return FunctionPipelineExecutor.compile(self._compiler())

    def to_generator(self):
        from ..executors.python import GeneratorPipelineExecutor

        return GeneratorPipelineExecutor.compile(self._compiler())

    def to_dask(self):
        from ..executors.dask import DaskPipelineExecutor

        return DaskPipelineExecutor.compile(self._compiler())

    def sha256(self):
        return dataclass_sha256(self, ignore_keys=self._hash_exclude_)


RecipeCompiler = Callable[[BaseRecipe], Pipeline]


@dataclass
class FilePatternMixin:
    file_pattern: FilePattern

    def copy_pruned(self, nkeep: int = 2):
        """Make a copy of this recipe with a pruned file pattern.
        :param nkeep: The number of items to keep from each ConcatDim sequence.
        """

        new_pattern = prune_pattern(self.file_pattern, nkeep=nkeep)
        return replace(self, file_pattern=new_pattern)


@dataclass
class StorageMixin:
    """Provides the storage configuration for Pangeo Forge recipe classes.
    :param storage_config: The storage configuration.
    """

    storage_config: StorageConfig = field(default_factory=temporary_storage_config)

    @property
    def target(self):
        return f"{self.storage_config.target.fs.protocol}://{self.storage_config.target.root_path}"

    @property
    def target_mapper(self):
        return self.storage_config.target.get_mapper()

#########################################
# OpenMosaic Recipe Base Implementation #
#########################################

@dataclass
class BaseMosaicRecipe(BaseRecipe, StorageMixin, FilePatternMixin):
    # TODO everything
    ...


@dataclass
class CommonMosaicRecipe(BaseMosaicRecipe):
    # TODO everything
    _compiler = common_recipe_compiler

    @property
    def iter_times(self):
        # TODO
        ...

    @property
    def iter_radars_at_time(self):
        # TODO return {time: iterable} like dict
        ...

    def __getitem__(self, key: Hashable):
        # TODO subset for a particular time configuration
        # return MosaicSubrecipe(...)
        ...


@dataclass
class MosaicSubrecipe(BaseMosaicRecipe):
    # TODO: modifications on BaseMosaicRecipe to only function at a single time
    # Private API?
    # does this need a _compiler?
    ...


def common_recipe_compiler(recipe) -> Pipeline:
    stages = [
        Stage(name="scan_inputs", function=scan_inputs),
        Stage(
            name="process_times",
            function=Pipeline(
                stages=[
                    Stage(name="init_grid", function=init_grid),
                    Stage(
                        name="process_radars",
                        function=process_radar,
                        mappable=recipe.iter_radars_at_time
                    ),
                    Stage(name="aggregate_subgrids", function=aggregate_subgrids),
                    Stage(name="calculate_3d", function=calculate_3d),
                    Stage(name="finalize_3d", function=finalize_3d)
                ],
                config=recipe
            ),
            mappable=recipe.iter_times
        ),
        Stage(name="calculate_4d", function=calculate_4d),
        Stage(name="finalize_4d", function=finalize_4d)
    ],
    return Pipeline(stages=stages, config=recipe)


def scan_inputs(config: CommonMosaicRecipe):
    # TODO: prepare the iterables of radar volume targets and subgrids for each time
    ...


def calculate_4d(config: CommonMosaicRecipe):
    # TODO: apply "across time" calculations
    ...


def finalize_4d(config: CommonMosaicRecipe):
    # TODO: any culminating operations for the full mosaic across all times
    ...


def init_grid(config: MosaicSubrecipe):
    # TODO: set up the arrays for the 3D grid
    ...


def process_radar(radar_target: Any, config: MosaicSubrecipe):
    # TODO the following
    #    - open
    #    - adjust/normalize
    #    - coordinate compute
    #    - radial calculations
    #    - regrid to intermediate
    #    - local grid calculations
    #    - transform (i.e., coarsen, if needed) to subgrid and output
    # TODO: get better typing for radar_target
    ...


def aggregate_subgrids(config: MosaicSubrecipe):
    # TODO combine all subgrids to the full grid
    ...


def calculate_3d(config: MosaicSubrecipe):
    # TODO any 3D grid operations (like MESH/SHI, layer aggregation, echo tops)
    ...


def finalize_3d(config: MosaicSubrecipe):
    # TODO any final steps for each time (principally output)
    ...
