"""Compiled Waymo Open Dataset protobuf definitions.

The generated *_pb2.py files use ``from waymo_open_dataset import ...``
imports.  We alias this package as ``waymo_open_dataset`` in sys.modules
so those imports resolve without requiring the actual Waymo SDK.
"""

import importlib
import sys

# Make `from waymo_open_dataset import X` resolve to this package.
_self = sys.modules[__name__]
sys.modules.setdefault("waymo_open_dataset", _self)
sys.modules.setdefault(
    "waymo_open_dataset.protos",
    importlib.import_module("rigid_flow.data.waymo_protos.protos"),
)
