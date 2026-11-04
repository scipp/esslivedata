# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Transmission from area detector frames labelled by an image key log.

Each frame is classified by the image key valid at the frame's own timestamp,
following the NXdetector ``image_key`` convention (0 sample, 1 flat field /
open beam, 2 dark field; 3 invalid and 4 background are not used). Frames are
accumulated per type, and

    transmission = (sample - dark) / (open beam - dark)

is formed from the mean frame of each type, so phases of different duration
are comparable.

Limitations:

- A key change is attributed to frames by timestamp only. If the key log
  delivers a change after the batch holding the frames it applies to was
  processed, those frames stay in the previous type.
- Equal exposure time is assumed for the three types.
- Frames must keep their shape: a change (e.g. of binning) makes every batch
  fail until the job is reset.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import NewType

import sciline
import scipp as sc

from ..preprocessors.accumulators import NoCopyAccumulator
from ..preprocessors.frame_batch import FrameBatch
from .area_detector_view import make_logical_view
from .stream_processor_workflow import StreamProcessorWorkflow


class ImageType(StrEnum):
    """Image types accumulated by the workflow."""

    SAMPLE = 'sample'
    OPEN_BEAM = 'open_beam'
    DARK = 'dark'


IMAGE_TYPE_FOR_KEY: dict[int, ImageType] = {
    0: ImageType.SAMPLE,
    1: ImageType.OPEN_BEAM,
    2: ImageType.DARK,
}
"""Image key values that are accumulated; any other value drops the frame."""

_NO_KEY = -1
"""Image key assigned to frames older than the first sample of the key log."""

ImageKeyLog = NewType('ImageKeyLog', sc.DataArray)
"""NXlog-like history of the image key (context)."""

FrameTransform = NewType('FrameTransform', Callable[[sc.DataArray], sc.DataArray])
"""Applied to each frame before accumulation, e.g. downsampling."""


@dataclass(frozen=True)
class ClassifiedFrames:
    """Result of classifying one batch.

    All three image types are always present, empty ones zero, so that
    results of different batches can be added.
    """

    sums: sc.DataArray
    counts: sc.DataArray
    dropped: sc.DataArray


CumulativeSums = NewType('CumulativeSums', sc.DataArray)
CumulativeCounts = NewType('CumulativeCounts', sc.DataArray)
DroppedFrames = NewType('DroppedFrames', sc.DataArray)
SampleImage = NewType('SampleImage', sc.DataArray | None)
OpenBeamImage = NewType('OpenBeamImage', sc.DataArray | None)
DarkImage = NewType('DarkImage', sc.DataArray | None)
Transmission = NewType('Transmission', sc.DataArray | None)


def _image_keys(key_log: sc.DataArray, times: sc.Variable) -> list[int]:
    """Image key valid at each time, :data:`_NO_KEY` before the first sample."""
    values = key_log.data
    if values.dtype in (sc.DType.float32, sc.DType.float64):
        values = sc.round(values)
    log = sc.DataArray(
        values.to(dtype='int64'), coords={'time': key_log.coords['time']}
    )
    lookup = sc.lookup(
        log, 'time', mode='previous', fill_value=sc.scalar(_NO_KEY, unit=values.unit)
    )
    return lookup[times].values.tolist()


def classify_frames(
    batch: FrameBatch, key_log: ImageKeyLog, transform: FrameTransform
) -> ClassifiedFrames:
    """Sum the frames of a batch per image type, by the key valid at each frame."""
    keys = _image_keys(key_log, batch.times)
    sums: dict[ImageType, sc.DataArray] = {}
    counts = dict.fromkeys(ImageType, 0)
    dropped = 0
    for key, frame in zip(keys, batch.frames, strict=True):
        image_type = IMAGE_TYPE_FOR_KEY.get(key)
        if image_type is None:
            dropped += 1
            continue
        image = transform(frame).to(dtype='float64', copy=True)
        if image_type in sums:
            sums[image_type] += image
        else:
            sums[image_type] = image
        counts[image_type] += 1
    template = (
        next(iter(sums.values()))
        if sums
        else transform(batch.frames[0]).to(dtype='float64')
    )
    zeros = sc.zeros_like(template)
    types = sc.array(dims=['image_type'], values=[t.value for t in ImageType])
    return ClassifiedFrames(
        sums=sc.concat(
            [sums.get(image_type, zeros) for image_type in ImageType], 'image_type'
        ).assign_coords(image_type=types),
        counts=sc.DataArray(
            sc.array(
                dims=['image_type'],
                values=[counts[t] for t in ImageType],
                unit='counts',
                dtype='int64',
            ),
            coords={'image_type': types},
        ),
        dropped=sc.DataArray(sc.scalar(dropped, unit='counts', dtype='int64')),
    )


def cumulative_sums(classified: ClassifiedFrames) -> CumulativeSums:
    return CumulativeSums(classified.sums)


def cumulative_counts(classified: ClassifiedFrames) -> CumulativeCounts:
    return CumulativeCounts(classified.counts)


def dropped_frames(classified: ClassifiedFrames) -> DroppedFrames:
    return DroppedFrames(classified.dropped)


def _mean_image(
    sums: sc.DataArray, counts: sc.DataArray, image_type: ImageType
) -> sc.DataArray | None:
    i = list(ImageType).index(image_type)
    n = counts['image_type', i].value
    if n == 0:
        return None
    return (sums['image_type', i] / sc.scalar(float(n))).drop_coords('image_type')


def sample_image(sums: CumulativeSums, counts: CumulativeCounts) -> SampleImage:
    return SampleImage(_mean_image(sums, counts, ImageType.SAMPLE))


def open_beam_image(sums: CumulativeSums, counts: CumulativeCounts) -> OpenBeamImage:
    return OpenBeamImage(_mean_image(sums, counts, ImageType.OPEN_BEAM))


def dark_image(sums: CumulativeSums, counts: CumulativeCounts) -> DarkImage:
    return DarkImage(_mean_image(sums, counts, ImageType.DARK))


def transmission(
    sample: SampleImage, open_beam: OpenBeamImage, dark: DarkImage
) -> Transmission:
    """(sample - dark) / (open beam - dark), NaN where open beam <= dark."""
    if sample is None or open_beam is None or dark is None:
        return Transmission(None)
    attenuation = open_beam - dark
    result = (sample - dark) / attenuation
    result.data = sc.where(
        attenuation.data > sc.scalar(0.0, unit=attenuation.unit),
        result.data,
        sc.scalar(float('nan'), unit=result.unit),
    )
    return Transmission(result)


def build_transmission_workflow(
    *, source_name: str, transform: Callable[[sc.DataArray], sc.DataArray]
) -> StreamProcessorWorkflow:
    """
    Transmission workflow for the area detector ``source_name``.

    The image key log is a context input: the routing layer binds the stream
    carrying it to :data:`ImageKeyLog`.

    Parameters
    ----------
    source_name:
        Area detector providing the frames.
    transform:
        Applied to every frame before accumulation. Use it to downsample.
    """
    workflow = sciline.Pipeline(
        (
            classify_frames,
            cumulative_sums,
            cumulative_counts,
            dropped_frames,
            sample_image,
            open_beam_image,
            dark_image,
            transmission,
        )
    )
    workflow[FrameTransform] = transform
    outputs = {
        'sample': SampleImage,
        'open_beam': OpenBeamImage,
        'dark': DarkImage,
        'transmission': Transmission,
        'dropped_frames': DroppedFrames,
    }
    return StreamProcessorWorkflow(
        workflow,
        dynamic_keys={source_name: FrameBatch},
        target_keys=outputs,
        accumulators={
            CumulativeSums: NoCopyAccumulator(),
            CumulativeCounts: NoCopyAccumulator(),
            DroppedFrames: NoCopyAccumulator(),
        },
        publish_on_change=outputs,
    )


def transmission_factory(
    *,
    transform: Callable[[sc.DataArray, str], sc.DataArray] | None = None,
    reduction_dim: str | list[str] | None = None,
) -> Callable[[str], StreamProcessorWorkflow]:
    """
    Create a workflow factory for :func:`build_transmission_workflow`.

    Takes the same arguments as
    :meth:`~ess.livedata.workflows.area_detector_view.AreaDetectorView.view_factory`,
    so one instrument transform serves both the image view and the transmission.

    Parameters
    ----------
    transform:
        Signature ``(da, source_name) -> da``, e.g. a fold for downsampling. If
        None, frames are used as they are.
    reduction_dim:
        Dimension(s) to sum over after applying ``transform``.
    """

    def factory(source_name: str) -> StreamProcessorWorkflow:
        return build_transmission_workflow(
            source_name=source_name,
            transform=make_logical_view(source_name, transform, reduction_dim),
        )

    return factory
