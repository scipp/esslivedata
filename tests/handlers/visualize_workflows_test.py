# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from typing import NewType

import graphviz
import sciline

from ess.livedata.handlers.stream_processor_workflow import StreamProcessorWorkflow

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
