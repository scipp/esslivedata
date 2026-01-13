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

import sciline
import scipp as sc
from scippnexus import NXdetector

from ess.reduce.nexus.types import EmptyDetector, Filename, NeXusName, SampleRun


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


class DetectorDataSource:
    """
    Base class for detector data source configuration.

    Subclasses define how EmptyDetector is obtained for the workflow.
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
        raise NotImplementedError


class NeXusDetectorSource(DetectorDataSource):
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


class DetectorNumberSource(DetectorDataSource):
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
