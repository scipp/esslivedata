# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
from structlog.testing import capture_logs

from ess.livedata.config.detector_downsampling import resolve_downsampling


def square_grid(side: int, first_id: int = 0) -> sc.Variable:
    return sc.arange(
        'detector_number', first_id, first_id + side * side, unit=None
    ).fold(dim='detector_number', sizes={'dim_0': side, 'dim_1': side})


def resolve(
    declared: sc.Variable | None,
    *,
    resolution: int = 512,
    source_resolution: int = 4096,
    reconfigurable: bool = True,
):
    return resolve_downsampling(
        'det', resolution, source_resolution, reconfigurable, declared
    )


class TestTargetGrid:
    def test_is_a_square_zero_based_enumeration_of_the_target_resolution(self) -> None:
        # Ids the preprocessor produces are 0-based indices into this grid.
        grid = resolve(square_grid(4096)).grid
        assert grid.sizes == {'dim_0': 512, 'dim_1': 512}
        assert grid.min().value == 0
        assert grid.max().value == 512 * 512 - 1

    def test_does_not_depend_on_the_declared_grid(self) -> None:
        assert sc.identical(resolve(square_grid(4096)).grid, resolve(None).grid)

    def test_keeps_the_declared_dimension_names(self) -> None:
        # The coarse grid is the same detector, so an instrument that names its
        # axes gets them back and needs no rename in its view transform.
        declared = sc.arange('detector_number', 1024 * 1024, unit=None).fold(
            dim='detector_number', sizes={'x': 1024, 'y': 1024}
        )
        assert resolve(declared).grid.sizes == {'x': 512, 'y': 512}


class TestIdBase:
    def test_is_taken_from_the_declared_grid(self) -> None:
        # Detectors disagree on where they start counting, and assuming wrong
        # corrupts the inferred source resolution rather than merely shifting
        # the image, so the base is read rather than assumed.
        assert resolve(square_grid(4096, first_id=1)).first_id == 1

    def test_is_assumed_zero_with_a_warning_when_no_file_was_read(self) -> None:
        with capture_logs() as logs:
            downsampling = resolve(None)

        assert downsampling.first_id == 0
        assert any(e['event'] == 'downsampling_without_geometry' for e in logs)


class TestNonPowerOfTwoResolutions:
    """Only the ratio has to be a power of two, not the resolutions."""

    def test_accepts_a_panel_and_target_that_are_not_powers_of_two(self) -> None:
        downsampling = resolve(
            square_grid(1000), resolution=250, source_resolution=4000
        )
        assert downsampling.resolution == 250
        assert downsampling.grid.sizes == {'dim_0': 250, 'dim_1': 250}


class TestFixedSourceResolution:
    """Where the readout cannot be reconfigured, the stride is stated."""

    def test_is_taken_from_the_configuration(self) -> None:
        downsampling = resolve(
            square_grid(1024), source_resolution=1024, reconfigurable=False
        )
        assert downsampling.source_resolution == 1024

    def test_is_none_where_the_readout_is_reconfigurable(self) -> None:
        assert resolve(square_grid(1024)).source_resolution is None

    def test_rejects_a_declared_side_that_disagrees(self) -> None:
        # Nothing revises a fixed stride at runtime, so the disagreement has to
        # be loud here rather than a quietly scrambled image later. A smaller
        # side is merely a past readout where the resolution is inferred.
        with pytest.raises(ValueError, match='fixed'):
            resolve(square_grid(1024), source_resolution=4096, reconfigurable=False)


class TestDeclaredGridRejections:
    """The layout the remap assumes, checked while the file is in memory."""

    def test_rejects_a_non_square_grid(self) -> None:
        declared = sc.arange('detector_number', 4096 * 2048, unit=None).fold(
            dim='detector_number', sizes={'dim_0': 4096, 'dim_1': 2048}
        )
        with pytest.raises(ValueError, match='square'):
            resolve(declared)

    def test_rejects_a_side_the_target_does_not_tile(self) -> None:
        with pytest.raises(ValueError, match='power of two'):
            resolve(square_grid(1000))

    def test_rejects_a_side_tiled_by_a_ratio_that_is_not_a_power_of_two(self) -> None:
        # 1000 = 200 * 5 tiles cleanly, but 200 doubled never reaches 1000, so
        # the preprocessor could not infer this source however well it tiles.
        with pytest.raises(ValueError, match='power of two'):
            resolve(square_grid(1000), resolution=200, source_resolution=1600)

    def test_rejects_a_side_above_the_configured_maximum(self) -> None:
        # source_resolution is meant to be what the hardware can read out, so a
        # file declaring more than that means one of the two is wrong.
        with pytest.raises(ValueError, match='at most'):
            resolve(square_grid(4096), source_resolution=2048)

    def test_rejects_ids_that_are_not_row_major_contiguous(self) -> None:
        # The remap inverts 'id = x * side + y'. A file enumerating the grid in
        # any other order would scramble the image with nothing to notice it by.
        with pytest.raises(ValueError, match='row-major'):
            resolve(square_grid(1024).transpose().copy())

    def test_rejects_ids_with_gaps(self) -> None:
        declared = square_grid(1024).copy()
        declared.values[3, 7] += 1_000_000
        with pytest.raises(ValueError, match='row-major'):
            resolve(declared)
