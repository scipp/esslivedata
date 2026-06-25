# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Projection of designated workflow outputs onto the NICOS device topic.

A NICOS *derived device* is a workflow output designated in the per-instrument
:class:`~ess.livedata.config.device_contract.DeviceContract`. The
:class:`Projector` selects the contracted outputs from each job result and
republishes them on a dedicated, low-volume Kafka topic keyed by a stable
*device name* — free of the ``job_number`` carried by the main data path — so
NICOS sees a stable device identity across reconfigurations. Which outputs are
eligible is decided once, when the contract is loaded; the projector trusts the
contract and projects whatever it designates.

Each projected message carries a 0-D ``generation_token`` coordinate (the job's
creation time in nanoseconds since the epoch), letting consumers detect when the
underlying job was replaced by a reconfigure.
"""

from __future__ import annotations

import scipp as sc

from ..config.device_contract import DeviceContract
from .job import JobResult
from .message import Message, StreamId, StreamKind
from .timestamp import Timestamp

GENERATION_TOKEN_COORD = 'generation_token'  # noqa: S105
"""Coordinate name carrying the per-job generation token (ns since epoch)."""


class Projector:
    """Builds NICOS device-projection messages from job results.

    Parameters
    ----------
    device_contract:
        The per-instrument contract deciding which
        ``(workflow_id, source_name, output_name)`` outputs are devices and
        under which device name they are exposed.
    """

    def __init__(self, *, device_contract: DeviceContract) -> None:
        self._device_contract = device_contract

    def project(self, results: list[JobResult]) -> list[Message[sc.DataArray]]:
        """Project the contracted outputs of the given results.

        The generation token is read from each result's ``generation_token``
        field (stamped by the JobManager at production time), so a finishing
        job's final result is projected even though its registry entry has
        already been removed. A result without a token is skipped.

        Parameters
        ----------
        results:
            Valid job results, each carrying a ``DataGroup`` of named outputs.

        Returns
        -------
        :
            One message per projected output, keyed by device name on the
            :attr:`~ess.livedata.core.message.StreamKind.LIVEDATA_PROJECTION`
            stream, with the generation token attached as a 0-D coordinate.
        """
        messages: list[Message[sc.DataArray]] = []
        for result in results:
            if result.data is None or result.generation_token is None:
                continue
            token = sc.scalar(result.generation_token.to_ns(), unit='ns')
            for output_name, da in result.data.items():
                device_name = self._device_contract.device_name(
                    result.workflow_id, result.job_id.source_name, output_name
                )
                if device_name is None:
                    continue
                messages.append(
                    Message(
                        timestamp=result.start_time or Timestamp.from_ns(0),
                        stream=StreamId(
                            kind=StreamKind.LIVEDATA_PROJECTION, name=device_name
                        ),
                        value=da.assign_coords({GENERATION_TOKEN_COORD: token}),
                    )
                )
        return messages
