# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
from typing import NewType

import pytest
import sciline

from ess.livedata.handlers.stream_processor_workflow import StreamProcessorWorkflow

Streamed = NewType('Streamed', int)
Context = NewType('Context', int)
Static = NewType('Static', int)
ProcessedContext = NewType('ProcessedContext', int)
ProcessedStreamed = NewType('ProcessedStreamed', int)
Output = NewType('Output', int)


@pytest.fixture
def base_workflow_with_context() -> sciline.Pipeline:
    def make_static() -> Static:
        make_static.call_count += 1
        return Static(2)

    make_static.call_count = 0

    def process_context(context: Context, static: Static) -> ProcessedContext:
        process_context.call_count += 1
        return ProcessedContext(context * static)

    process_context.call_count = 0

    def process_streamed(
        streamed: Streamed, context: ProcessedContext
    ) -> ProcessedStreamed:
        return ProcessedStreamed(streamed + context)

    def finalize(streamed: ProcessedStreamed) -> Output:
        return Output(streamed)

    return sciline.Pipeline((make_static, process_context, process_streamed, finalize))


@pytest.fixture
def base_workflow_no_context() -> sciline.Pipeline:
    def make_static() -> Static:
        return Static(2)

    def process_streamed_direct(
        streamed: Streamed, static: Static
    ) -> ProcessedStreamed:
        return ProcessedStreamed(streamed + static)

    def finalize(streamed: ProcessedStreamed) -> Output:
        return Output(streamed)

    return sciline.Pipeline((make_static, process_streamed_direct, finalize))


class TestStreamProcessorWorkflow:
    def test_workflow_initialization(self, base_workflow_with_context):
        """Test that StreamProcessorWorkflow can be initialized correctly."""
        workflow = StreamProcessorWorkflow(
            base_workflow_with_context,
            dynamic_keys={'streamed': Streamed},
            context_keys={'context': Context},
            target_keys={'output': Output},
            accumulators=(ProcessedStreamed,),
        )
        assert workflow is not None

    def test_accumulate_and_finalize(self, base_workflow_with_context):
        """Test the basic accumulate and finalize workflow."""
        workflow = StreamProcessorWorkflow(
            base_workflow_with_context,
            dynamic_keys={'streamed': Streamed},
            context_keys={'context': Context},
            target_keys={'output': Output},
            accumulators=(ProcessedStreamed,),
        )

        # Set context data
        workflow.accumulate({'context': Context(5)}, start_time=1000, end_time=2000)

        # Accumulate dynamic data
        workflow.accumulate({'streamed': Streamed(10)}, start_time=1000, end_time=2000)
        workflow.accumulate({'streamed': Streamed(20)}, start_time=1000, end_time=2000)

        # Finalize and check result
        result = workflow.finalize()
        # Accumulated as 10 + 20 = 30, then 30 + 10 = 40, then 40 + 10 = 50
        assert result == {'output': Output(50)}

    def test_clear_workflow(self, base_workflow_with_context):
        """Test that clearing the workflow resets its state."""
        workflow = StreamProcessorWorkflow(
            base_workflow_with_context,
            dynamic_keys={'streamed': Streamed},
            context_keys={'context': Context},
            target_keys={'output': Output},
            accumulators=(ProcessedStreamed,),
        )

        # Accumulate some data
        workflow.accumulate({'context': Context(5)}, start_time=1000, end_time=2000)
        workflow.accumulate({'streamed': Streamed(10)}, start_time=1000, end_time=2000)

        # Clear and start fresh
        workflow.clear()

        # Set new context and data
        workflow.accumulate({'context': Context(2)}, start_time=1000, end_time=2000)
        workflow.accumulate({'streamed': Streamed(15)}, start_time=1000, end_time=2000)

        result = workflow.finalize()
        # Expected: context (2) * static (2) = 4, streamed: 15, final: 15 + 4 = 19
        assert result == {'output': Output(19)}

    def test_partial_data_accumulation(self, base_workflow_with_context):
        """Test accumulating data with only some keys present."""
        workflow = StreamProcessorWorkflow(
            base_workflow_with_context,
            dynamic_keys={'streamed': Streamed},
            context_keys={'context': Context},
            target_keys={'output': Output},
            accumulators=(ProcessedStreamed,),
        )

        # Accumulate with only context
        workflow.accumulate({'context': Context(3)}, start_time=1000, end_time=2000)

        # Accumulate with only streamed data
        workflow.accumulate({'streamed': Streamed(7)}, start_time=1000, end_time=2000)

        # Accumulate with unknown keys (should be ignored)
        workflow.accumulate({'unknown': 42}, start_time=1000, end_time=2000)

        result = workflow.finalize()
        # Expected: context (3) * static (2) = 6, streamed: 7, final: 7 + 6 = 13
        assert result == {'output': Output(13)}

    def test_target_keys_with_simplified_names(self, base_workflow_with_context):
        """Test initialization with simplified output names."""
        workflow = StreamProcessorWorkflow(
            base_workflow_with_context,
            dynamic_keys={'streamed': Streamed},
            context_keys={'context': Context},
            target_keys={'simplified_output': Output},
            accumulators=(ProcessedStreamed,),
        )

        workflow.accumulate({'context': Context(4)}, start_time=1000, end_time=2000)
        workflow.accumulate({'streamed': Streamed(5)}, start_time=1000, end_time=2000)

        result = workflow.finalize()
        # Expected: context (4) * static (2) = 8, streamed: 5, final: 5 + 8 = 13
        # Simplified name is used as key
        assert result == {'simplified_output': Output(13)}

    def test_no_context_keys(self, base_workflow_no_context):
        """Test workflow without context keys."""
        workflow = StreamProcessorWorkflow(
            base_workflow_no_context,
            dynamic_keys={'streamed': Streamed},
            target_keys={'output': Output},
            accumulators=(ProcessedStreamed,),
        )

        # Only accumulate dynamic data
        workflow.accumulate({'streamed': Streamed(25)}, start_time=1000, end_time=2000)

        result = workflow.finalize()
        # Expected: streamed (25) + static (2) = 27
        assert result == {'output': Output(27)}
