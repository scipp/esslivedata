# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
"""
Configuration loader for transport settings.

This module provides Pydantic models and loading functions for configuring
how different stream kinds are transported (Kafka vs HTTP).
"""

from __future__ import annotations

from importlib import resources
from typing import Literal

import yaml
from pydantic import BaseModel

from ess.livedata.core.message import StreamKind


class StreamTransportConfig(BaseModel):
    """
    Configuration for transport of a specific stream kind.

    Parameters
    ----------
    kind:
        The stream kind to configure transport for.
    transport:
        Transport type to use ('kafka' or 'http').
    url:
        HTTP base URL (required for http transport, unused for kafka).
    """

    kind: StreamKind
    transport: Literal['kafka', 'http']
    url: str | None = None


class TransportConfig(BaseModel):
    """
    Transport configuration for all streams.

    Parameters
    ----------
    streams:
        List of transport configurations for each stream kind.
    """

    streams: list[StreamTransportConfig]


def load_transport_config(instrument: str) -> TransportConfig:
    """
    Load transport configuration for the specified instrument.

    Parameters
    ----------
    instrument:
        Instrument name (e.g. 'dummy', 'dream', 'bifrost').

    Returns
    -------
    :
        Validated transport configuration.

    Raises
    ------
    FileNotFoundError:
        If the transport config file for the instrument does not exist.
    """
    config_path = resources.files('ess.livedata.config.transports')
    config_file = f'{instrument}.yaml'

    try:
        with config_path.joinpath(config_file).open() as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Transport config file '{config_file}' not found for instrument "
            f"'{instrument}' in config/transports/"
        ) from None

    return TransportConfig(**config_data)
