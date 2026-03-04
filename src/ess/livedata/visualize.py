# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Visualization utilities for registered workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import graphviz
import structlog

from .config.instrument import Instrument
from .config.workflow_spec import WorkflowConfig, WorkflowId

logger = structlog.get_logger(__name__)


def visualize_workflows(
    instrument: Instrument,
    *,
    output_dir: Path | None = None,
    format: str = "svg",
    **kwargs: Any,
) -> dict[str, graphviz.Digraph]:
    """Visualize all registered workflows for an instrument.

    Iterates all workflow specs registered with the instrument, instantiates each
    with default parameters, and collects visualizations from workflows that support
    it (i.e., those backed by a ``StreamProcessor``).

    Parameters
    ----------
    instrument:
        Instrument with factories loaded (call ``instrument.load_factories()`` first).
    output_dir:
        If provided, render each graph to ``{output_dir}/{workflow_id}.{format}``.
    format:
        Output format when rendering to files (e.g., "svg", "png", "pdf").
    **kwargs:
        Passed to ``StreamProcessorWorkflow.visualize()``. See
        :py:meth:`ess.reduce.streaming.StreamProcessor.visualize` for options.

    Returns
    -------
    :
        Mapping from workflow ID string to graphviz Digraph.
    """
    factory = instrument.workflow_factory
    graphs: dict[str, graphviz.Digraph] = {}

    for workflow_id, spec in factory.items():
        workflow = _try_create_workflow(factory, workflow_id, spec)
        if workflow is None:
            continue
        if not hasattr(workflow, 'visualize'):
            continue

        try:
            graph = workflow.visualize(**kwargs)
        except Exception as e:
            logger.warning(
                "Failed to visualize workflow",
                workflow_id=str(workflow_id),
                error=str(e),
            )
            continue

        key = str(workflow_id)
        graphs[key] = graph

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for key, graph in graphs.items():
            filename = key.replace("/", "_")
            graph.render(output_dir / filename, format=format, cleanup=True)

    return graphs


def _try_create_workflow(
    factory: Any,
    workflow_id: WorkflowId,
    spec: Any,
) -> Any | None:
    source_name = spec.source_names[0] if spec.source_names else ""
    config = WorkflowConfig(identifier=workflow_id, params={})
    try:
        return factory.create(source_name=source_name, config=config)
    except Exception as e:
        logger.warning(
            "Failed to create workflow", workflow_id=str(workflow_id), error=str(e)
        )
        return None
