# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
TBL workflow spec registration.
"""

import scipp as sc

from ess.livedata.config import (
    AxisRange,
    Instrument,
    SourceMetadata,
    filter_authorized_streams,
    instrument_registry,
    name_streams,
)
from ess.livedata.config.device_contract import DETECTOR_VIEW_DEVICES
from ess.livedata.config.workflow_spec import DETECTORS
from ess.livedata.workflows.detector_view_specs import (
    AreaDetectorViewOutputs,
    SpectrumViewSpec,
)
from ess.livedata.workflows.monitor_workflow_specs import (
    register_monitor_workflow_specs,
)

from .streams_parsed import PARSED_STREAMS
from .views import (
    get_he3_detector_view,
    get_he3_spectrum,
    get_multiblade_spectrum,
    get_multiblade_view,
    identity,
    name_image_dims,
)

detector_names = [
    'timepix3',
    'multiblade',
    'he3_bank_0',
    'he3_bank_1',
    'ngem',
    # not listing orca since it does not have (and does not need) detector numbers
]

monitor_names = ['beam_monitor_1']

#: Side length the Timepix3 panel is ingested at; see name_image_dims.
TIMEPIX3_IMAGE_RESOLUTION = 512
#: Largest grid the Timepix3 panel can read out. The readout is
#: reconfigured during operation, so this bounds the inferred streamed
#: resolution rather than stating it.
TIMEPIX3_PANEL_RESOLUTION = 4096

instrument = Instrument(
    name='tbl',
    detector_names=detector_names,
    monitors=monitor_names,
    # Bandwidth choppers feeding the wavelength-LUT cascade, in beam order
    # (source -> sample): bwc_1 sits at 8.54 m from the source, bwc_2 at 8.56 m.
    choppers=['bwc_1', 'bwc_2'],
    # beam_monitor_1 rides filter_changer_3, whose travel is not known. The axis
    # is perpendicular to the beam, 9.1 m from the source, so the monitor's
    # flight path hardly depends on it: +-1 m of travel widens its lookup-table
    # block by 4 cm. The range is therefore declared generously.
    axis_ranges={
        '/entry/instrument/filter_changer_3/value': AxisRange(
            lower=sc.scalar(-1.0, unit='m'), upper=sc.scalar(1.0, unit='m')
        )
    },
    streams=name_streams(filter_authorized_streams(PARSED_STREAMS)),
    source_metadata={
        'timepix3': SourceMetadata(title='Timepix3'),
        'multiblade': SourceMetadata(title='Multiblade'),
        'he3_bank_0': SourceMetadata(title='He3 Bank 0'),
        'he3_bank_1': SourceMetadata(title='He3 Bank 1'),
        'ngem': SourceMetadata(title='nGEM'),
        'orca': SourceMetadata(title='ORCA'),
        'beam_monitor_1': SourceMetadata(title='Beam Monitor'),
    },
)

instrument_registry.register(instrument)

#: Positions along the beam relative to the sample, in metres, as the geometry
#: file states them. Wavelength mode computes flight paths from these, but they
#: are unverified and mostly look like placeholders (no NICOS device reports
#: them yet, see #1356). The view descriptions show them to the user; a test
#: keeps this in sync with the geometry file.
GEOMETRY_FILE_Z_POSITIONS = {
    'source': -15.0,
    # Static part of the chain only: the monitor rides the live filter_changer_3
    # axis, which moves it along x and so leaves z unchanged.
    'beam_monitor_1': -5.89,
    'he3_bank_0': 1.0,
    'he3_bank_1': 2.0,
    'multiblade': 1.0,
    'ngem': -1.0,
}


def _unverified_positions_warning(source_names: list[str]) -> str:
    positions = ', '.join(
        f'{instrument.source_metadata[name].title} at '
        f'{GEOMETRY_FILE_Z_POSITIONS[name]:+.1f} m'
        for name in source_names
    )
    return (
        '<b>Warning:</b> wavelength mode uses unverified positions from the'
        f' geometry file: {positions}, source at'
        f" {GEOMETRY_FILE_Z_POSITIONS['source']:+.1f} m. Positions are along"
        ' the beam relative to the sample. Treat wavelengths as approximate.'
    )


register_monitor_workflow_specs(
    instrument,
    monitor_names,
    extra_description=_unverified_positions_warning(monitor_names),
)

instrument.configure_detector_downsampling(
    'timepix3',
    resolution=TIMEPIX3_IMAGE_RESOLUTION,
    source_resolution=TIMEPIX3_PANEL_RESOLUTION,
    reconfigurable=True,
)

instrument.add_logical_view(
    name='tbl_detector_timepix3',
    title='Timepix3 Detector',
    description=(
        f'{TIMEPIX3_IMAGE_RESOLUTION}x{TIMEPIX3_IMAGE_RESOLUTION} image,'
        ' downsampled from the streamed resolution as the events are'
        ' ingested. The streamed resolution is read from the event ids; a'
        ' readout reconfiguration restarts the cumulative image, since counts'
        ' taken before and after it are not commensurable.'
    ),
    source_names=['timepix3'],
    transform=name_image_dims,
    # No wavelength mode: the geometry file holds positions for the full panel
    # resolution, not for the downsampled grid the events are ingested at.
    roi_support=True,
    device_outputs=DETECTOR_VIEW_DEVICES,
)

instrument.add_logical_view(
    name='multiblade_detector_view',
    title='Multiblade Detector',
    description=(
        'Counts folded into blade, wire, and strip dimensions<br><br>'
        + _unverified_positions_warning(['multiblade'])
    ),
    source_names=['multiblade'],
    wavelength=True,
    transform=get_multiblade_view,
    # ROI geometries are rectangles and polygons on a 2D screen; this view is 3D.
    roi_support=False,
    output_ndim=3,
    spectrum_view=SpectrumViewSpec(
        transform=get_multiblade_spectrum,
        output_dims=['blade', 'wire'],
        extra_description='Summed across strips, yielding per-blade, per-wire spectra.',
    ),
    device_outputs=DETECTOR_VIEW_DEVICES,
)

instrument.add_logical_view(
    name='he3_detector_view',
    title='He3 Detector',
    description=(
        'Combined view of both detector banks with tube and pixel axes<br><br>'
        + _unverified_positions_warning(['he3_bank_0', 'he3_bank_1'])
    ),
    source_names=['he3_bank_0', 'he3_bank_1'],
    wavelength=True,
    transform=get_he3_detector_view,
    roi_support=True,
    spectrum_view=SpectrumViewSpec(
        transform=get_he3_spectrum,
        output_dims=['tube', 'pixel'],
        extra_description='Per-tube, per-pixel spectra, without spatial reduction.',
    ),
    device_outputs=DETECTOR_VIEW_DEVICES,
)

instrument.add_logical_view(
    name='ngem_detector_view',
    title='NGEM Detector',
    description=(
        '2D detector counts view<br><br>' + _unverified_positions_warning(['ngem'])
    ),
    source_names=['ngem'],
    wavelength=True,
    transform=identity,
    reduction_dim='dim_0',
    roi_support=True,
    device_outputs=DETECTOR_VIEW_DEVICES,
)

orca_view_handle = instrument.register_spec(
    group=DETECTORS,
    name='tbl_area_detector_orca',
    version=1,
    title='Orca Detector',
    description='512x512 image downsampled from full resolution',
    source_names=['orca'],
    # AreaDetectorView renders images only; it neither reads ROI requests nor
    # publishes readbacks, so it must not declare the ROI outputs.
    outputs=AreaDetectorViewOutputs,
    device_outputs=DETECTOR_VIEW_DEVICES,
)
