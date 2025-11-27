# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
from streaming_data_types import area_detector_ad00

# Scipp only supports: bool, float32, float64, int32, int64, string, datetime64
_DTYPE_MAP = {
    np.dtype('uint8'): np.int32,
    np.dtype('int8'): np.int32,
    np.dtype('uint16'): np.int32,
    np.dtype('int16'): np.int32,
    np.dtype('uint32'): np.int64,
    np.dtype('uint64'): np.float64,  # May lose precision for large values
}


def ad00_to_scipp(ad00: area_detector_ad00.ADArray) -> sc.DataArray:
    """
    Convert an ad00 ADArray to a scipp DataArray.

    Parameters
    ----------
    ad00:
        The area detector data from the ad00 schema.

    Returns
    -------
    :
        A scipp DataArray with the image data.

    Notes
    -----
    Scipp only supports bool, float32, float64, int32, int64, string, and datetime64.
    Unsigned integer types and smaller integer types are converted to compatible types:
    - uint8, int8, uint16, int16 -> int32
    - uint32 -> int64
    - uint64 -> float64 (may lose precision for large values)
    """
    # Create dimension names based on the number of dimensions
    dims = [f'dim_{i}' for i in range(len(ad00.dimensions))]
    # Reshape data according to dimensions (ad00.dimensions gives the shape)
    data = ad00.data.reshape(tuple(ad00.dimensions))

    # Convert unsupported dtypes to scipp-compatible types
    if data.dtype in _DTYPE_MAP:
        data = data.astype(_DTYPE_MAP[data.dtype])

    return sc.DataArray(data=sc.array(dims=dims, values=data))
