# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Shared detector transforms for use across instruments.

These transforms handle common detector types that appear at multiple beamlines.
"""

from collections.abc import Callable

import scipp as sc


def make_image_fold_transform(
    input_dims: tuple[str, str],
    output_dims: tuple[str, str] = ('x', 'y'),
    target_size: int = 512,
) -> Callable[[sc.DataArray], sc.DataArray]:
    """
    Create a transform that renames and folds 2D image dimensions.

    Parameters
    ----------
    input_dims:
        Names of the input dimensions to rename (x_dim, y_dim).
    output_dims:
        Names for the output dimensions after renaming.
    target_size:
        Target size for each dimension after folding. The original dimension
        is split into (target_size, original_size // target_size).

    Returns
    -------
    :
        Transform function that folds a DataArray to the target resolution.
    """

    def transform(da: sc.DataArray) -> sc.DataArray:
        da = da.rename_dims(
            {input_dims[0]: output_dims[0], input_dims[1]: output_dims[1]}
        )
        da = da.fold(
            dim=output_dims[0],
            sizes={output_dims[0]: target_size, f'{output_dims[0]}_bin': -1},
        )
        da = da.fold(
            dim=output_dims[1],
            sizes={output_dims[1]: target_size, f'{output_dims[1]}_bin': -1},
        )
        return da

    return transform


# Timepix3 detector: 4096x4096 downsampled to 512x512
# Dimension names come from NeXus geometry with pixel offset coordinates
timepix3_fold = make_image_fold_transform(
    input_dims=('x_pixel_offset', 'y_pixel_offset'),
    target_size=512,
)

# Orca area detector (ad00 schema): 2048x2048 downsampled to 512x512
# Uses generic dimension names from the ad00 message format
orca_fold = make_image_fold_transform(
    input_dims=('dim_0', 'dim_1'),
    target_size=512,
)
