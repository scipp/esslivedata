# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""
This config is used to setup live raw detector views.
Currently the instrument specific config is stored in python files, but
they can be moved to a separate file format in the future.
"""

import importlib
import pkgutil
from types import ModuleType


def available_instruments() -> list[str]:
    """Get list of available instruments based on config modules."""
    return [
        name
        for _, name, _ in pkgutil.iter_modules(__path__)
        if name != '__init__' and not name.startswith('_')
    ]


def get_config(instrument: str) -> ModuleType:
    """Get config module for given instrument."""
    try:
        return importlib.import_module(f'.{instrument}', __package__)
    except (ImportError, AttributeError) as e:
        raise ValueError(f'No config found for instrument {instrument}') from e


__all__ = ['available_instruments', 'get_config']
