# Copyright (c) 2022 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Base class(es) for defining an OpenMosaic recipe, modeled after pangeo-forge.

This file contains code originating from pangeo-forge-recipes, (c) 2022 pangeo-forge-recipes
developers. Used with modification under the terms of the Apache-2.0 license (see LICENSE for
full text of license). Modifications (to-date) entail docstring/comment modifications, changes
to imports (based on differing APIs), and deletion of unsupported methods. Otherwise, all
functional code that is present is taken as-is from pangeo-forge-recipes.

TODO: update docstrings to NumPy conventions as with remainder of OpenMosaic
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field, replace
from typing import Callable, ClassVar

from ..executors.base import Pipeline
from ..patterns import FilePattern, prune_pattern
from ..core.serialization import dataclass_sha256
from ..storage import StorageConfig, temporary_storage_config


@dataclass
class BaseRecipe(ABC):
    """Base class for recipes.

    Reused from pangeo-forge-recipes, with deletion of `to_prefect` and `to_beam` methods,
    which are not supported by OpenMosaic at this time.
    """
    _compiler: ClassVar[RecipeCompiler]
    _hash_exclude_ = ["storage_config"]

    def to_function(self):
        from ..executors import FunctionPipelineExecutor

        return FunctionPipelineExecutor.compile(self._compiler())

    def to_generator(self):
        from ..executors import GeneratorPipelineExecutor

        return GeneratorPipelineExecutor.compile(self._compiler())

    def to_dask(self):
        from pangeo_forge_recipes.executors import DaskPipelineExecutor

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
