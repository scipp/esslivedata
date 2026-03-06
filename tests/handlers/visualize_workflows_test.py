# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from typing import NewType

import graphviz
import sciline

from ess.livedata.handlers.stream_processor_workflow import StreamProcessorWorkflow
from ess.livedata.visualize import visualize_workflows

Input = NewType('Input', int)
Output = NewType('Output', int)


def _identity(x: Input) -> Output:
    return Output(x)


class TestStreamProcessorWorkflowVisualize:
    def test_returns_graphviz_digraph(self):
        workflow = StreamProcessorWorkflow(
            sciline.Pipeline((_identity,)),
            dynamic_keys={'input': Input},
            target_keys={'output': Output},
            accumulators=(Output,),
        )
        graph = workflow.visualize()
        assert isinstance(graph, graphviz.Digraph)

    def test_passes_kwargs_through(self):
        workflow = StreamProcessorWorkflow(
            sciline.Pipeline((_identity,)),
            dynamic_keys={'input': Input},
            target_keys={'output': Output},
            accumulators=(Output,),
        )
        graph = workflow.visualize(compact=True, show_legend=False)
        assert isinstance(graph, graphviz.Digraph)


class TestVisualizeWorkflows:
    def test_dummy_instrument_returns_graphs(self):
        from ess.livedata.config.instruments.dummy.specs import instrument

        instrument.load_factories()
        graphs = visualize_workflows(instrument)
        assert len(graphs) > 0
        for graph in graphs.values():
            assert isinstance(graph, graphviz.Digraph)

    def test_dummy_instrument_includes_total_counts(self):
        from ess.livedata.config.instruments.dummy.specs import instrument

        instrument.load_factories()
        graphs = visualize_workflows(instrument)
        keys = list(graphs.keys())
        assert any('total_counts' in k for k in keys)

    def test_renders_to_output_dir(self, tmp_path):
        from ess.livedata.config.instruments.dummy.specs import instrument

        instrument.load_factories()
        graphs = visualize_workflows(instrument, output_dir=tmp_path, format="svg")
        assert len(graphs) > 0
        svg_files = list(tmp_path.glob("*.svg"))
        assert len(svg_files) == len(graphs)

    def test_skips_non_visualizable_workflows(self):
        """Workflows without a visualize method are silently skipped."""
        from ess.livedata.config.instrument import Instrument

        instrument = Instrument(name='test_viz')
        handle = instrument.register_spec(
            name='no_viz',
            version=1,
            title='No viz',
            source_names=['src'],
            outputs=_make_dummy_outputs(),
        )

        class _PlainWorkflow:
            def accumulate(self, data, *, start_time, end_time):
                pass

            def finalize(self):
                return {}

            def clear(self):
                pass

        handle.attach_factory()(lambda: _PlainWorkflow())

        graphs = visualize_workflows(instrument)
        assert graphs == {}


def _make_dummy_outputs():
    from pydantic import BaseModel, ConfigDict

    class DummyOutputs(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        result: int = 0

    return DummyOutputs
