# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
"""Common workflows that are used by multiple instruments."""

from __future__ import annotations

from typing import Any

import scipp as sc

from ess.livedata.handlers.accumulators import LogData
from ess.livedata.handlers.to_nxlog import ToNXlog
from ess.reduce import streaming


class TimeseriesAccumulator(streaming.Accumulator[sc.DataArray]):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Use ToNXlog which is used as a preprocessor elsewhere (for raw f144). The
        # interface is similar but not identical, so we wrap instead of inheriting.
        self._to_nxlog: ToNXlog | None = None

    @property
    def is_empty(self) -> bool:
        return self._to_nxlog is None

    def _get_value(self) -> sc.DataArray:
        if self._to_nxlog is None:
            raise ValueError("No data accumulated")
        return self._to_nxlog.get()

    def _do_push(self, value: sc.DataArray) -> None:
        if self._to_nxlog is None:
            self._to_nxlog = ToNXlog(
                attrs={'units': str(value.unit)}, data_dims=tuple(value.dims)
            )
        self._to_nxlog.add(
            0,
            LogData(
                time=value.coords['time'].value,
                value=value.values,
                variances=value.variances,
            ),
        )

    def clear(self) -> None:
        if self._to_nxlog is not None:
            self._to_nxlog.clear()
