# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
TBL instrument stream mapping configuration.
"""

from ess.livedata.config.env import StreamingEnv
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping

from .._ess import make_common_stream_mapping_inputs, make_dev_stream_mapping

# Note: Panel size is fake and does not correspond to production setting
detector_fakes = {
    'timepix3_detector': (1, 4096**2),
    'he3_detector_bank0': (1, 400),  # 4x100
    'he3_detector_bank1': (401, 800),  # 4x100
    'ngem_detector': (1, 2 * 128**2),
    'multiblade_detector': (1, 14 * 64 * 32),
}

# Area detector fakes: detector_name -> (height, width) shape
# Actual size is 2048x2048 but that would require higher message size limits for dev
area_detector_fakes = {'orca_detector': (512, 512)}


def _make_tbl_detectors() -> StreamLUT:
    """
    TBL detector mapping.

    Input keys based on
    https://confluence.ess.eu/display/ECDC/Kafka+Topics+Overview+for+Instruments
    """
    return {
        InputStreamKey(
            topic='tbl_detector_tpx3', source_name='timepix3'
        ): 'timepix3_detector',
        InputStreamKey(
            topic='tbl_detector_mb', source_name='multiblade'
        ): 'multiblade_detector',
        InputStreamKey(
            topic='tbl_detector_3he', source_name='bank0'
        ): 'he3_detector_bank0',
        InputStreamKey(
            topic='tbl_detector_3he', source_name='bank1'
        ): 'he3_detector_bank1',
        InputStreamKey(
            topic='tbl_detector_ngem', source_name='tbl-ngem'
        ): 'ngem_detector',
    }


def _make_tbl_area_detectors() -> StreamLUT:
    """TBL area detector mapping for ad00 schema detectors."""
    return {
        InputStreamKey(
            topic='tbl_area_detector_orca', source_name='hama_kfk1'
        ): 'orca_detector',
    }


stream_mapping = {
    StreamingEnv.DEV: make_dev_stream_mapping(
        'tbl',
        detector_names=list(detector_fakes),
        area_detector_names=list(area_detector_fakes),
    ),
    StreamingEnv.PROD: StreamMapping(
        **make_common_stream_mapping_inputs(instrument='tbl'),
        detectors=_make_tbl_detectors(),
        area_detectors=_make_tbl_area_detectors(),
    ),
}
