# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
from ess.estia import EstiaWorkflow
from ess.reduce.nexus.types import Filename, Position, SampleRun
from scippnexus import NXsource

from ess.livedata.config.instruments.estia.live_geometry import (
    insert_live_geometry,
)
from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename
from ess.livedata.workflows.lut_ranges import component_ltotal_range

pytestmark = pytest.mark.slow


def test_source_and_monitor_distances_come_from_geometry_artifact():
    filename = get_nexus_geometry_filename('estia')
    workflow = EstiaWorkflow()
    workflow[Filename[SampleRun]] = filename
    insert_live_geometry(workflow, str(filename))
    source = workflow.compute(Position[NXsource, SampleRun])
    assert sc.isclose(sc.norm(source).to(unit='m'), sc.scalar(35.0, unit='m'))
    start, stop = component_ltotal_range(filename, 'beam_monitor', is_monitor=True)
    assert sc.isclose((start + stop) / 2, sc.scalar(10.68, unit='m'))
