#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Migrate dashboard config files from 4-part to 3-part WorkflowId format.

PR #901 dropped the ``namespace`` segment from ``WorkflowId``: stringified IDs
changed from ``instrument/namespace/name/version`` to ``instrument/name/version``.
The dashboard persists configs under ``$LIVEDATA_CONFIG_DIR/{instrument}/*.yaml``
(or, by default, the platform-specific user config dir). Files written before
PR #901 hold 4-part strings; the relevant ones are:

* ``plot_configs.yaml`` - workflow_ids embedded in layer ``data_sources``
* ``workflow_configs.yaml`` - top-level dict keyed by stringified WorkflowId

Other files are processed too: this script walks every YAML in each
instrument's config dir and rewrites every 4-part WorkflowId-shaped string,
whether it appears as a value or as a dict key. Idempotent: 3-part IDs are
left alone.

Each modified file is backed up to ``<file>.bak`` before being overwritten via
atomic rename.

Usage::

    python scripts/migrate_workflow_ids.py            # all instruments
    python scripts/migrate_workflow_ids.py --instrument dummy
    python scripts/migrate_workflow_ids.py --config-dir /custom/dir
    python scripts/migrate_workflow_ids.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

import platformdirs
import yaml

# Matches a 4-part WorkflowId string ``instrument/namespace/name/version`` with
# integer version. Restrictive enough that ordinary topic/source names won't
# match (they don't usually contain three slashes followed by a digit-only tail).
_OLD_WORKFLOW_ID = re.compile(r'^([^/]+)/([^/]+)/([^/]+)/(\d+)$')


def _migrate_id(value: str) -> str | None:
    """Return the 3-part form of a 4-part WorkflowId, or None otherwise."""
    match = _OLD_WORKFLOW_ID.match(value)
    if match is None:
        return None
    instrument, _namespace, name, version = match.groups()
    return f'{instrument}/{name}/{version}'


def _migrate(value: Any, counter: list[int]) -> Any:
    """Recursively rewrite 4-part WorkflowId strings in keys and values."""
    if isinstance(value, dict):
        new: dict[Any, Any] = {}
        for k, v in value.items():
            new_k = k
            if isinstance(k, str):
                if (migrated := _migrate_id(k)) is not None:
                    counter[0] += 1
                    new_k = migrated
            new[new_k] = _migrate(v, counter)
        return new
    if isinstance(value, list):
        return [_migrate(v, counter) for v in value]
    if isinstance(value, str):
        if (migrated := _migrate_id(value)) is not None:
            counter[0] += 1
            return migrated
    return value


def migrate_file(path: Path, *, dry_run: bool) -> int:
    """Migrate a single YAML file. Returns the number of rewritten IDs."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        return 0
    counter = [0]
    new_data = _migrate(data, counter)
    if counter[0] == 0:
        return 0
    if dry_run:
        return counter[0]
    backup = path.with_suffix(path.suffix + '.bak')
    backup.write_bytes(path.read_bytes())
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        yaml.safe_dump(new_data, f, default_flow_style=False, sort_keys=False)
    tmp.replace(path)
    return counter[0]


def _resolve_base_dir(override: Path | None) -> Path:
    if override is not None:
        return override
    if (env_dir := os.environ.get('LIVEDATA_CONFIG_DIR')) is not None:
        return Path(env_dir)
    return Path(platformdirs.user_config_dir('esslivedata', appauthor=False))


def _instrument_dirs(base: Path, instrument: str | None) -> list[Path]:
    if instrument is not None:
        d = base / instrument
        return [d] if d.is_dir() else []
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--instrument',
        help='Migrate only this instrument; default migrates every '
        'instrument subdirectory under the base config dir.',
    )
    parser.add_argument(
        '--config-dir',
        type=Path,
        help='Override base config dir. Defaults to LIVEDATA_CONFIG_DIR or '
        'the platform user config dir, matching the dashboard.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Report changes without writing.',
    )
    args = parser.parse_args()

    base = _resolve_base_dir(args.config_dir)
    dirs = _instrument_dirs(base, args.instrument)
    if not dirs:
        target = f'{base}/{args.instrument}' if args.instrument else str(base)
        print(f'No instrument config dirs found under {target}', file=sys.stderr)
        return 0

    total = 0
    for inst_dir in dirs:
        for yaml_path in sorted(inst_dir.glob('*.yaml')):
            n = migrate_file(yaml_path, dry_run=args.dry_run)
            if n:
                action = 'would migrate' if args.dry_run else 'migrated'
                print(f'{action} {n} workflow_id(s) in {yaml_path}')
                total += n
    suffix = ' (dry-run)' if args.dry_run else ''
    print(f'Done: {total} workflow_id(s) updated{suffix}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
