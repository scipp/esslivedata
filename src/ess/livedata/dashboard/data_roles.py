# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Constants for data source roles in plot configurations.

These constants define the keys used in PlotConfig.data_sources to identify
the role of each data source:

- PRIMARY: The main data source for plotting/histogramming
- X_AXIS: X-axis correlation data for correlation histograms
- Y_AXIS: Y-axis correlation data for 2D correlation histograms
"""

PRIMARY = 'primary'
X_AXIS = 'x_axis'
Y_AXIS = 'y_axis'
