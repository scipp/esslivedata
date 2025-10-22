# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ODIN instrument factory implementations.
"""

from ess.livedata.config import Instrument

from . import specs


def setup_factories(instrument: Instrument) -> None:
    """Initialize ODIN-specific factories and workflows."""
    import h5py
    import scipp as sc

    from ess.livedata.handlers.detector_data_handler import DetectorProjection

    # Patch the Odin geometry file with:
    # 1. Non-zero z (needed for detector xy projection)
    # 2. Axes names and mapping to detector number shape, since ScippNexus
    #    cannot infer these automatically from the Timepix3 data.
    # Note: We do this every time on import. Accessing `instrument.nexus_file`
    # the first time will actually fetch the file using pooch, so it reverts
    # this change every time.
    with h5py.File(instrument.nexus_file, 'r+') as f:
        det = f['entry/instrument/event_mode_detectors/timepix3']
        trans = det['transformations/translation']
        trans[...] = 1.0
        det.attrs['axes'] = ['x_pixel_offset', 'y_pixel_offset']
        det.attrs['detector_number_indices'] = [0, 1]

    # Configure detector with custom group name
    instrument.configure_detector(
        'timepix3', detector_group_name='event_mode_detectors'
    )

    # Create detector projection
    _xy_projection = DetectorProjection(
        instrument=instrument,
        projection='xy_plane',
        resolution={'timepix3': {'y': 512, 'x': 512}},
    )

    def _resize_image(da: sc.DataArray) -> sc.DataArray:
        from ess.imaging.tools import resample

        # 2048*2048 is the actual panel size, and 1024*1024 in the test file,
        # but ess.livedata might not be able to keep up with that
        # so we resample to 128*128 ((1024/8) * (1024/8)) for now.
        return resample(
            da, sizes={'x_pixel_offset': 8, 'y_pixel_offset': 8}, method='sum'
        )

    # Detector view configuration
    from ess.livedata.handlers.detector_data_handler import DetectorLogicalView

    _panel_0_view = DetectorLogicalView(instrument=instrument, transform=_resize_image)

    specs.panel_0_view_handle.attach_factory()(_panel_0_view.make_view)
