# Copyright (c) 2022 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""Class(es) for in-Python-only pipeline executors for OpenMosaic, modeled after pangeo-forge.

This file contains code originating from pangeo-forge-recipes, (c) 2022 pangeo-forge-recipes
developers. Used with modification under the terms of the Apache-2.0 license (see LICENSE for
full text of license). Modifications (to-date) entail docstring/comment modifications only,
all functional code is taken as-is from pangeo-forge-recipes.

TODO: update docstrings to NumPy conventions as with remainder of OpenMosaic
"""

from __future__ import annotations

from typing import Any, Callable, Generator

from .base import Pipeline, PipelineExecutor

GeneratorPipeline = Generator[Any, None, None]


class GeneratorPipelineExecutor(PipelineExecutor[GeneratorPipeline]):
    """An executor which returns a Generator.
    The Generator yeilds `function, args, kwargs`, which can be called step by step
    to iterate through the recipe.
    """

    @staticmethod
    def compile(pipeline: Pipeline):
        def generator_function():
            for stage in pipeline.stages:
                if stage.mappable is not None:
                    for m in stage.mappable:
                        yield stage.function, (m,), dict(config=pipeline.config)
                else:
                    yield stage.function, (), dict(config=pipeline.config)

        return generator_function()

    @staticmethod
    def execute(generator: GeneratorPipeline) -> None:
        for func, args, kwargs in generator:
            func(*args, **kwargs)


class FunctionPipelineExecutor(PipelineExecutor[Callable]):
    """A generator which returns a single callable python function with no
    arguments. Calling this function will run the whole recipe"""

    @staticmethod
    def compile(pipeline: Pipeline):
        def function():
            for stage in pipeline.stages:
                if stage.mappable is not None:
                    for m in stage.mappable:
                        stage.function(m, config=pipeline.config)
                else:
                    stage.function(config=pipeline.config)

        return function

    @staticmethod
    def execute(func: Callable) -> None:
        func()
