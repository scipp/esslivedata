# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Sciline key aliases for preprocessor-to-workflow input types.

Instrument workflow factories use these aliases in ``dynamic_keys`` instead
of spelling out the concrete Sciline key.  When the preprocessor changes
(e.g. switching between raw events and pre-grouped events), only the alias
definition here needs updating — not every instrument factory.
"""

from ess.reduce.nexus.types import RawDetector, SampleRun

PreprocessedDetectorEvents = RawDetector[SampleRun]
