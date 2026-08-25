# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Guards the committed UI config fixtures against workflow/output schema drift.

These fixtures (``tests/dashboard/ui_config_fixtures/<instrument>/``) seed the
dashboard via ``--config-dir`` so UI-test runs skip clicking through source
selection and plot-grid setup. When workflows or their outputs are renamed the
fixtures go stale; these tests fail loudly so they get regenerated.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from ess.livedata.config import instrument_registry
from ess.livedata.config.instruments import get_config
from ess.livedata.config.workflow_spec import WorkflowId
from ess.livedata.dashboard.plotter_registry import plotter_registry

FIXTURES_DIR = Path(__file__).parent / 'ui_config_fixtures'
INSTRUMENTS = [p.name for p in FIXTURES_DIR.iterdir() if p.is_dir()]


def _load(instrument: str, name: str) -> dict:
    with open(FIXTURES_DIR / instrument / f'{name}.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture(params=INSTRUMENTS)
def instrument(request: pytest.FixtureRequest) -> str:
    name = request.param
    get_config(name)  # registers the instrument's workflows
    return name


def _registry(instrument: str):
    return instrument_registry[instrument].workflow_factory


class TestWorkflowConfigsFixture:
    def test_workflow_ids_exist(self, instrument: str) -> None:
        registry = _registry(instrument)
        for workflow_id_str in _load(instrument, 'workflow_configs'):
            assert WorkflowId.from_string(workflow_id_str) in registry

    def test_staged_sources_are_valid(self, instrument: str) -> None:
        registry = _registry(instrument)
        for workflow_id_str, entry in _load(instrument, 'workflow_configs').items():
            spec = registry[WorkflowId.from_string(workflow_id_str)]
            staged_sources = set(entry['jobs'])
            assert staged_sources <= set(spec.source_names)

    def test_no_runtime_state_persisted(self, instrument: str) -> None:
        # Fixtures must be reproducible: no runtime job UUIDs.
        for entry in _load(instrument, 'workflow_configs').values():
            assert set(entry) == {'jobs'}


def _layers(instrument: str) -> Iterator[dict]:
    """Every plot layer of the instrument's fixture, across grids and cells."""
    for grid in _load(instrument, 'plot_configs')['plot_grids']['grids']:
        for cell in grid['cells']:
            yield from cell['layers']


class TestPlotConfigsFixture:
    def test_plot_data_sources_resolve(self, instrument: str) -> None:
        registry = _registry(instrument)
        for layer in _layers(instrument):
            for ds in layer['data_sources'].values():
                spec = registry[WorkflowId.from_string(ds['workflow_id'])]
                view_names = {v.name for v in spec.get_output_views()}
                assert ds['view_name'] in view_names
                assert set(ds['source_names']) <= set(
                    spec.sources_for_output(ds['view_name'])
                )

    def test_plotters_accept_the_data_they_are_pointed_at(
        self, instrument: str
    ) -> None:
        # An incompatible pairing survives config load and only fails at
        # compute time, leaving a red cell instead of a plot; the browser
        # tests would still pass on the tab's other figures.
        registry = _registry(instrument)
        for layer in _layers(instrument):
            ds = layer['data_sources']['primary']
            spec = registry[WorkflowId.from_string(ds['workflow_id'])]
            template = spec.get_output_template(ds['view_name'])
            compatible = plotter_registry.get_compatible_plotters(
                {ds['view_name']: template}
            )
            assert layer['plot_name'] in compatible, (
                f"{layer['plot_name']} cannot plot view '{ds['view_name']}' of "
                f"{ds['workflow_id']} (dims={template.dims})"
            )

    def test_workflows_backing_plots_are_staged(self, instrument: str) -> None:
        staged = set(_load(instrument, 'workflow_configs'))
        for layer in _layers(instrument):
            for ds in layer['data_sources'].values():
                assert ds['workflow_id'] in staged
