#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Render workflow graphs for a given instrument."""

import argparse
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


def _load_instrument(name: str):
    from importlib import import_module

    from ess.livedata.config.instrument import instrument_registry

    import_module(f'ess.livedata.config.instruments.{name}.specs')
    instrument = instrument_registry[name]
    instrument.load_factories()
    return instrument


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instrument', required=True, help='Instrument name')
    parser.add_argument(
        '--workflow',
        metavar='ID',
        help='Workflow ID to render (see --list). Renders all if omitted.',
    )
    parser.add_argument('--format', default='svg', help='Output format (default: svg)')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('.'),
        help='Output directory (default: current directory)',
    )
    parser.add_argument(
        '--list', action='store_true', help='List workflow IDs and exit'
    )
    args = parser.parse_args()

    instrument = _load_instrument(args.instrument)
    factory = instrument.workflow_factory
    all_ids = sorted(str(wid) for wid in factory)

    if args.list:
        for wid in all_ids:
            print(wid)
        return

    from ess.livedata.config.workflow_spec import WorkflowId

    if args.workflow is not None:
        target = WorkflowId.from_string(args.workflow)
        if target not in factory:
            print(f"Unknown workflow: {args.workflow}", file=sys.stderr)
            print("Available workflows:", file=sys.stderr)
            for wid in all_ids:
                print(f"  {wid}", file=sys.stderr)
            sys.exit(1)
        items = [(target, factory[target])]
    else:
        items = list(factory.items())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for wid, spec in items:
        source_name = spec.source_names[0] if spec.source_names else ""
        from ess.livedata.config.workflow_spec import WorkflowConfig

        config = WorkflowConfig(identifier=wid, params={})
        try:
            workflow = factory.create(source_name=source_name, config=config)
        except Exception as e:
            logger.warning(
                "Failed to create workflow", workflow_id=str(wid), error=str(e)
            )
            continue
        if not hasattr(workflow, 'visualize'):
            continue
        try:
            graph = workflow.visualize()
        except Exception as e:
            logger.warning(
                "Failed to visualize workflow", workflow_id=str(wid), error=str(e)
            )
            continue
        filename = str(wid).replace('/', '_')
        graph.render(args.output_dir / filename, format=args.format, cleanup=True)
        print(f"Wrote {args.output_dir / filename}.{args.format}")
        rendered += 1

    if not rendered:
        print("No visualizable workflows found.", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
