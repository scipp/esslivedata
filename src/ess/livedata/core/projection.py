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

Devices are scalar *cumulative* outputs, which carry a 0-D ``start_time``
coordinate (stamped by the JobManager at production time, ns since epoch). It is
constant for the lifetime of a generation and changes on reset or reconfigure,
so NICOS uses it as a change-detector to distinguish a post-reset zero from a
genuine low reading.
"""

from __future__ import annotations

import scipp as sc

from ..config.device_contract import DeviceContract
from .job import JobResult
from .message import Message, StreamId, StreamKind
from .timestamp import Timestamp


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

        Parameters
        ----------
        results:
            Valid job results, each carrying a ``DataGroup`` of named outputs.

        Returns
        -------
        :
            One message per projected output, keyed by device name on the
            :attr:`~ess.livedata.core.message.StreamKind.LIVEDATA_PROJECTION`
            stream. The output's ``start_time`` coordinate rides along as the
            generation change-detector.
        """
        messages: list[Message[sc.DataArray]] = []
        for result in results:
            if result.data is None:
                continue
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
                        value=da,
                    )
                )
        return messages
