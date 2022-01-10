# Copyright (c) 2021 OpenMosaic Developers.
# Distributed under the terms of the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
"""...TODO..."""

from .core import LevelIILoader

class S3LevelIILoader(LevelIILoader):
    # TODO docstring
    def __init__(self, *args, **kwargs):
        # TODO docstring
        # TODO allow bucket/boto3 control, but enable smart defaults
        super().__init__(*args, **kwargs)

    """
    TODO
    
    - Define the valid files types to check for (and how they map to any needed preprocessing)
    - Scan for available files for site given time range (a higher level utility will use this
      to determine if there are enough sites present to have a meaningful mosaic)
    - Download a file from s3 to local cache (unzipping if gzipped)
    - Open source LII files to pyart radars (can hopefully reuse from super, possibly munging
      path)
    - Any extra downloaded file cache management (i.e., can delete the level II file once the
      processed cf-radial is saved.)
    - ...otherwise everything else should follow from the base loader...
    """