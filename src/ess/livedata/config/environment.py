# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Environment configuration for ESSlivedata services."""

import os

ENV_VAR = 'LIVEDATA_ENV'
DEFAULT_ENV = 'dev'
PRODUCTION_ENV = 'production'


def get_environment() -> str:
    """Get the current environment name, defaulting to 'dev'."""
    return os.getenv(ENV_VAR, DEFAULT_ENV)


def is_production() -> bool:
    """Check if running in production environment."""
    return get_environment() == PRODUCTION_ENV
