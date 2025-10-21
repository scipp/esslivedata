# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Bifrost spectrometer stream mapping configuration.
"""

from collections.abc import Generator

import scipp as sc

from ess.livedata.config.env import StreamingEnv
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping

from .._ess import make_common_stream_mapping_inputs, make_dev_stream_mapping
from .specs import monitor_names


def _bifrost_generator() -> Generator[tuple[str, tuple[int, int]]]:
    # BEWARE! There are gaps in the detector_number per bank, which would usually get
    # dropped when mapping to pixels. BUT we merge banks for Bifrost, before mapping to
    # pixels, so the generated fake events in the wrong bank will end up in the right
    # bank. As a consequence we do not lose any fake events, but the travel over Kafka
    # with the wrong source_name.
    start = 125
    ntube = 3
    for channel in range(1, 10):
        for arc in range(1, 6):
            # Note: Actual start is at base + 100 * (channel - 1), but we start earlier
            # to get consistent counts across all banks, relating to comment above.
            base = ntube * 900 * (arc - 1)
            yield (
                f'{start}_channel_{channel}_{arc}_triplet',
                (base + 1, base + 2700),
            )
            start += 4
        start += 1


detectors_config = {'fakes': dict(_bifrost_generator())}

# Detector number configuration
detector_number = sc.arange('detector_number', 1, 5 * 3 * 9 * 100 + 1, unit=None).fold(
    dim='detector_number', sizes={'arc': 5, 'tube': 3, 'channel': 9, 'pixel': 100}
)


def _make_bifrost_detectors() -> StreamLUT:
    """
    Bifrost detector mapping.

    Input keys based on
    https://confluence.ess.eu/display/ECDC/Kafka+Topics+Overview+for+Instruments
    """
    # Source names have the format `arc=[0-4];triplet=[0-8]`.
    return {
        InputStreamKey(
            topic='bifrost_detector', source_name=f'arc={arc};triplet={triplet}'
        ): f'arc{arc}_triplet{triplet}'
        for arc in range(5)
        for triplet in range(9)
    }


stream_mapping = {
    StreamingEnv.DEV: make_dev_stream_mapping(
        'bifrost',
        detector_names=list(detectors_config['fakes']),
        monitor_names=monitor_names,
    ),
    StreamingEnv.PROD: StreamMapping(
        **make_common_stream_mapping_inputs(
            instrument='bifrost', monitor_names=monitor_names
        ),
        detectors=_make_bifrost_detectors(),
    ),
}
