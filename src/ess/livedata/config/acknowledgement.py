# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Command acknowledgement model for frontend-backend communication.

This implements the acknowledgement pattern from the NICOS-ESSlivedata interface spec.
"""

from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, Field


class AcknowledgementResponse(StrEnum):
    """Response type for command acknowledgements."""

    ACK = "ACK"
    ERR = "ERR"


class CommandAcknowledgement(BaseModel):
    """
    Acknowledgement response for commands sent to backend services.

    The backend echoes the message_id from the original command, allowing
    the frontend to correlate responses with pending commands.
    """

    key: ClassVar[str] = "command_ack"

    message_id: str = Field(description="Echoed from the original command")
    device: str = Field(description="Target device/source that processed the command")
    response: AcknowledgementResponse = Field(
        description="ACK for success, ERR for failure"
    )
    message: str | None = Field(
        default=None, description="Error message if response is ERR"
    )
