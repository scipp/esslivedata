# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Detector data source abstractions for workflow configuration.

This module provides abstractions for different sources of detector geometry
(EmptyDetector), allowing workflows to be configured with either NeXus file
data or synthetic detector structures.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Protocol

import sciline
import scipp as sc
from ess.reduce.nexus.types import EmptyDetector, Filename, NeXusName, SampleRun
from scippnexus import NXdetector

if TYPE_CHECKING:
    from ess.livedata.config.instrument import Instrument


def create_empty_detector(detector_number: sc.Variable) -> sc.DataArray:
    """
    Create an EmptyDetector structure from detector_number.

    This allows creating the detector structure without reading a NeXus file,
    enabling faster workflow startup for logical views.

    Parameters
    ----------
    detector_number:
        Detector number array defining the pixel structure.

    Returns
    -------
    :
        DataArray with empty bins and detector_number coordinate,
        compatible with EmptyDetector[SampleRun].
    """
    begin = sc.zeros(sizes=detector_number.sizes, dtype='int64', unit=None)
    end = begin.copy()
    events = sc.DataArray(
        data=sc.empty(dims=['event'], shape=[0], dtype='float32', unit='counts'),
        coords={'event_time_offset': sc.empty(dims=['event'], shape=[0], unit='ns')},
    )
    return sc.DataArray(
        data=sc.bins(begin=begin, end=end, dim='event', data=events),
        coords={'detector_number': detector_number},
    )


class DetectorDataSource(Protocol):
    """
    Protocol for detector data source configuration.

    Implementations define how EmptyDetector is obtained for the workflow.
    EmptyDetector provides the detector pixel structure needed by
    GenericNeXusWorkflow to group events by pixel (NeXusData → RawDetector).
    """

    def configure_workflow(self, workflow: sciline.Pipeline, source_name: str) -> None:
        """
        Configure the workflow to provide EmptyDetector.

        Parameters
        ----------
        workflow:
            Sciline pipeline to configure.
        source_name:
            Name of the detector source.
        """
        ...


class NeXusDetectorSource:
    """
    Load EmptyDetector from a NeXus file.

    Use this for geometric projections that need pixel positions,
    or when full detector geometry is required.

    Parameters
    ----------
    filename:
        Path to the NeXus geometry file.
    """

    def __init__(self, filename: pathlib.Path) -> None:
        self._filename = filename

    def configure_workflow(self, workflow: sciline.Pipeline, source_name: str) -> None:
        workflow[Filename[SampleRun]] = self._filename
        workflow[NeXusName[NXdetector]] = source_name


class DetectorNumberSource:
    """
    Create EmptyDetector from detector_number without file I/O.

    Use this for logical views where only pixel structure is needed,
    enabling faster workflow startup. Also useful for testing.

    Parameters
    ----------
    detector_number:
        Detector number array defining the pixel structure.
    """

    def __init__(self, detector_number: sc.Variable) -> None:
        self._detector_number = detector_number

    def configure_workflow(self, workflow: sciline.Pipeline, source_name: str) -> None:
        workflow[EmptyDetector[SampleRun]] = create_empty_detector(
            self._detector_number
        )


class InstrumentDetectorSource:
    """
    Create EmptyDetector from an Instrument's configured detector_number.

    Use this for logical views where the detector_number is configured in the
    Instrument and may differ for each source_name. This enables fast startup
    without file I/O while supporting multiple detector sources.

    Parameters
    ----------
    instrument:
        The instrument configuration containing detector_number arrays.
    """

    def __init__(self, instrument: Instrument) -> None:
        self._instrument = instrument

    def configure_workflow(self, workflow: sciline.Pipeline, source_name: str) -> None:
        detector_number = self._instrument.get_detector_number(source_name)
        workflow[EmptyDetector[SampleRun]] = create_empty_detector(detector_number)
