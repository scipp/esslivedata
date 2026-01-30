# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-FileCopyrightText: 2025 Scipp contributors (https://github.com/scipp)
# SPDX-License-Identifier: BSD-3-Clause
from typing import NewType

import pytest
import sciline
import scipp as sc

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


# Types for window_outputs tests (need DataArray for assign_coords)
InputData = NewType('InputData', sc.DataArray)
CurrentOutput = NewType('CurrentOutput', sc.DataArray)
CumulativeOutput = NewType('CumulativeOutput', sc.DataArray)


class TestWindowOutputs:
    """Tests for window_outputs feature that adds time coords."""

    @pytest.fixture
    def dataarray_workflow(self) -> sciline.Pipeline:
        """Workflow that produces DataArray outputs."""

        def process_current(data: InputData) -> CurrentOutput:
            return CurrentOutput(data.copy())

        def process_cumulative(data: InputData) -> CumulativeOutput:
            return CumulativeOutput(data.copy())

        return sciline.Pipeline([process_current, process_cumulative])

    def test_window_output_has_time_coords_with_correct_values(
        self, dataarray_workflow
    ):
        """Test that window outputs have time coords with correct values."""
        from ess.reduce.streaming import EternalAccumulator

        workflow = StreamProcessorWorkflow(
            dataarray_workflow,
            dynamic_keys={'input': InputData},
            target_keys={'current': CurrentOutput, 'cumulative': CumulativeOutput},
            window_outputs=['current'],
            accumulators={
                CurrentOutput: EternalAccumulator(preprocess=None),
                CumulativeOutput: EternalAccumulator(preprocess=None),
            },
        )

        workflow.accumulate(
            {'input': sc.DataArray(sc.scalar(1.0))},
            start_time=1000,
            end_time=2000,
        )
        result = workflow.finalize()

        # Window output should have time coords with correct values
        assert result['current'].coords['time'].value == 1000
        assert result['current'].coords['start_time'].value == 1000
        assert result['current'].coords['end_time'].value == 2000
        assert result['current'].coords['time'].unit == 'ns'

        # Non-window output should not have time coords
        assert 'time' not in result['cumulative'].coords

    def test_time_coord_tracks_first_accumulate(self, dataarray_workflow):
        """Test that start_time uses first accumulate, end_time uses last."""
        from ess.reduce.streaming import EternalAccumulator

        workflow = StreamProcessorWorkflow(
            dataarray_workflow,
            dynamic_keys={'input': InputData},
            target_keys={'current': CurrentOutput},
            window_outputs=['current'],
            accumulators={CurrentOutput: EternalAccumulator(preprocess=None)},
        )

        # First accumulate
        workflow.accumulate(
            {'input': sc.DataArray(sc.scalar(1.0))},
            start_time=1000,
            end_time=2000,
        )
        # Second accumulate with different times
        workflow.accumulate(
            {'input': sc.DataArray(sc.scalar(2.0))},
            start_time=3000,
            end_time=4000,
        )
        result = workflow.finalize()

        # start_time from first accumulate, end_time from last
        assert result['current'].coords['start_time'].value == 1000
        assert result['current'].coords['end_time'].value == 4000

    def test_time_tracking_resets_after_finalize_and_clear(self, dataarray_workflow):
        """Test that time tracking resets after finalize() and clear()."""
        from ess.reduce.streaming import EternalAccumulator

        workflow = StreamProcessorWorkflow(
            dataarray_workflow,
            dynamic_keys={'input': InputData},
            target_keys={'current': CurrentOutput},
            window_outputs=['current'],
            accumulators={CurrentOutput: EternalAccumulator(preprocess=None)},
        )

        # First period
        workflow.accumulate(
            {'input': sc.DataArray(sc.scalar(1.0))},
            start_time=1000,
            end_time=2000,
        )
        result1 = workflow.finalize()
        assert result1['current'].coords['start_time'].value == 1000

        # After finalize, time should reset
        workflow.accumulate(
            {'input': sc.DataArray(sc.scalar(1.0))},
            start_time=5000,
            end_time=6000,
        )
        result2 = workflow.finalize()
        assert result2['current'].coords['start_time'].value == 5000

        # After clear, time should also reset
        workflow.accumulate(
            {'input': sc.DataArray(sc.scalar(1.0))},
            start_time=7000,
            end_time=8000,
        )
        workflow.clear()
        workflow.accumulate(
            {'input': sc.DataArray(sc.scalar(1.0))},
            start_time=9000,
            end_time=10000,
        )
        result3 = workflow.finalize()
        assert result3['current'].coords['start_time'].value == 9000
