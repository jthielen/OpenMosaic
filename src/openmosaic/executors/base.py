# Copyright (c) 2022 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Base class(es) for defining a pipeline executor for OpenMosaic, modeled after pangeo-forge.

This file contains code originating from pangeo-forge-recipes, (c) 2022 pangeo-forge-recipes
developers. Used with modification under the terms of the Apache-2.0 license (see LICENSE for
full text of license). Modifications (to-date) entail docstring/comment modifications only,
all functional code is taken as-is from pangeo-forge-recipes.

TODO: update docstrings to NumPy conventions as with remainder of OpenMosaic
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterable, Optional, TypeVar, Union

from mypy_extensions import NamedArg

Config = Any  # TODO: better typing for config
SingleArgumentStageFunction = Callable[[Any, NamedArg(type=Any, name="config")], None]  # noqa: F821
NoArgumentStageFunction = Callable[[NamedArg(type=Any, name="config")], None]  # noqa: F821
StageFunction = Union[NoArgumentStageFunction, SingleArgumentStageFunction]


class StageAnnotationType(enum.Enum):
    CONCURRENCY = enum.auto()
    RETRIES = enum.auto()


StageAnnotations = Dict[StageAnnotationType, Any]


@dataclass(frozen=True)
class Stage:
    function: StageFunction
    name: str
    mappable: Optional[Iterable] = None
    annotations: Optional[StageAnnotations] = None


@dataclass(frozen=True)
class Pipeline:
    stages: Iterable[Stage]
    config: Optional[Config] = None


T = TypeVar("T")


class PipelineExecutor(Generic[T]):
    @staticmethod
    def compile(pipeline: Pipeline) -> T:
        raise NotImplementedError

    @staticmethod
    def execute(plan: T):
        raise NotImplementedError
