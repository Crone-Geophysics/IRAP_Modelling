import re
from pathlib import Path

import matplotlib
from natsort import natsorted
import pandas as pd
import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (QLabel, QFormLayout, QWidget, QCheckBox, QFrame, QHBoxLayout, QSpinBox, QSizePolicy)
from matplotlib.pyplot import cm


class MUNTab(QWidget):
    toggle_sig = QtCore.pyqtSignal()

    def __init__(self, parent=None, color=None, axes=None, component=None):
        super().__init__()
        self.layout = QFormLayout()
        self.setLayout(self.layout)

        self.plot_cbox = QCheckBox("Plot")
        self.plot_cbox.setFocusPolicy(QtCore.Qt.NoFocus)
        self.plot_cbox.setChecked(True)

        self.layout.addRow(self.plot_cbox)
        self.layout.addRow("File Type", QLabel("MUN File"))

        # Channel selection frame
        self.ch_select_frame = QFrame()
        self.ch_select_frame.setLayout(QHBoxLayout())
        self.ch_select_frame.layout().setContentsMargins(0, 0, 0, 0)
        self.ch_select_frame.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.min_ch = QSpinBox()
        self.max_ch = QSpinBox()
        self.min_ch.setMinimum(1)
        self.max_ch.setMinimum(1)
        # self.min_ch.setFocusPolicy(QtCore.Qt.TabFocus)
        # self.max_ch.setFocusPolicy(QtCore.Qt.TabFocus)
        self.ch_select_frame.layout().addWidget(QLabel("Plot Channels"))
        self.ch_select_frame.layout().addWidget(self.min_ch)
        self.ch_select_frame.layout().addWidget(QLabel("to"))
        self.ch_select_frame.layout().addWidget(self.max_ch)

        self.file = None
        self.parent = parent
        self.color = color
        self.alpha = None  # Remember the last alpha
        self.color_by_channel = False
        self.ax = axes[component]
        self.component = component

        self.artists = []
        self.data = pd.DataFrame()

        # Signals
        self.plot_cbox.toggled.connect(self.toggle)
        self.min_ch.valueChanged.connect(lambda: self.update_channels("min"))
        self.max_ch.valueChanged.connect(lambda: self.update_channels("max"))

    def read(self, filepath):
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        ext = filepath.suffix.lower()

        if ext == '.dat':
            parser = MUNFile()
            try:
                file = parser.parse(filepath)
            except Exception as e:
                raise Exception(f"The following error occurred trying to parse the file: {e}.")
        else:
            raise ValueError(f"{ext} is not yet supported.")

        if file is None:
            raise ValueError(F"No data found in {filepath.name}.")

        # Add the file name as the default for the name in the legend
        self.layout.addRow('Units', QLabel(file.units))
        self.layout.addRow('Component', QLabel(self.component))

        self.layout.addRow(self.ch_select_frame)
        # Create a data frame with channel times and channel widths
        file.ch_times.index += 1
        file.ch_times.name = 'Times'
        self.layout.addRow('Channel Times', QLabel(file.ch_times.to_string()))

        # Set the channel range spin boxes
        self.min_ch.blockSignals(True)
        self.max_ch.blockSignals(True)
        self.min_ch.setValue(1)
        self.min_ch.setMaximum(len(file.ch_times))
        self.max_ch.setMaximum(len(file.ch_times))
        self.max_ch.setValue(len(file.ch_times))
        self.min_ch.blockSignals(False)
        self.max_ch.blockSignals(False)

        self.data = file.data
        self.file = file

    def plot(self, alpha=None, color_by_channel=None):
        """
        Plot the data on a mpl axes
        :param alpha: float
        :param color_by_channel: bool, color each channel a different color or color each line with self.color.
        """
        # Remove existing plotted lines
        self.remove()

        # Use the current alpha is none is passed
        if alpha is None:
            alpha = self.alpha
        else:
            self.alpha = alpha

        # Use the current legend coloring if none is passed
        if color_by_channel is None:
            color_by_channel = self.color_by_channel
        else:
            self.color_by_channel = color_by_channel

        self.artists = []

        channels = [f'{num}' for num in range(1, len(self.file.ch_times) + 1)]
        plotting_channels = channels[self.min_ch.value() - 1: self.max_ch.value()]

        data = self.data

        if data.empty:
            print(f"No {self.component} data in {self.file.filepath.name}.")
            return

        size = 8  # For scatter point size

        if color_by_channel is True:
            rainbow_color = iter(cm.gist_rainbow(np.linspace(0, 1, len(plotting_channels))))

        for ind, ch in enumerate(plotting_channels):
            # If coloring by channel, uses the rainbow color iterator and the label is the channel number.
            if color_by_channel is True:
                c = next(rainbow_color)  # Cycles through colors
                label = f"CH{ch} ({self.file.ch_times[int(ch)]} ms)"
            # If coloring by line, uses the tab's color, and the label is the file name.
            else:
                c = self.color
                if ind == 0:
                    label = f"{self.file.filepath.name}"
                else:
                    label = None

            x = data.Station.astype(float)
            y = data.loc[:, ch].astype(float)

            if len(x) == 1:
                style = 'o'
                artist = self.ax.scatter(x, y,
                                         color=c,
                                         marker=style,
                                         s=size,
                                         alpha=alpha,
                                         label=label)

            else:
                # style = '--' if 'Q' in freq else '-'
                artist, = self.ax.plot(x, y,
                                       color=c,
                                       alpha=alpha,
                                       # lw=count / 100,
                                       label=label)

            self.artists.append(artist)

            size += 2

    def remove(self):
        # Remove existing plotted lines
        if self.ax.lines:
            if all([artist in self.ax.lines for artist in self.artists]):
                for artist in self.artists:
                    self.ax.lines.remove(artist)
        if self.ax.collections:
            if all([artist in self.ax.collections for artist in self.artists]):
                for artist in self.artists:
                    self.ax.collections.remove(artist)

    def toggle(self):
        """Toggle the visibility of plotted lines/points"""

        lines = list(filter(lambda x: isinstance(x, matplotlib.lines.Line2D), self.artists))
        points = list(filter(lambda x: isinstance(x, matplotlib.collections.PathCollection), self.artists))  # Scatters

        if lines:
            if self.plot_cbox.isChecked():
                if all([a in self.ax.lines for a in lines]):  # If the lines are already plotted, pass.
                    pass
                else:
                    for artist in self.artists:
                        self.ax.lines.append(artist)
            else:
                for artist in self.artists:
                    self.ax.lines.remove(artist)

        if points:
            # Add or remove the scatter points
            if self.plot_cbox.isChecked():
                if all([a in self.ax.collections for a in points]):  # If the points are already plotted, pass.
                    pass
                else:
                    for artist in self.artists:
                        self.ax.collections.append(artist)
            else:
                for artist in self.artists:
                    self.ax.collections.remove(artist)

        self.toggle_sig.emit()

    def update_channels(self, which):
        """
        Change the plotted channel range
        :param which: str, which channel extreme was changed, min or max
        """
        self.min_ch.blockSignals(True)
        self.max_ch.blockSignals(True)

        min_ch = self.min_ch.value()
        max_ch = self.max_ch.value()

        if which == 'min':
            if min_ch > max_ch:
                self.max_ch.setValue(min_ch)

        else:
            if max_ch < min_ch:
                self.min_ch.setValue(max_ch)

        self.plot(alpha=None, color_by_channel=None)
        self.toggle_sig.emit()  # Updates the legend and re-draws

        # Re-scale the plots
        self.ax.relim()
        self.ax.autoscale()

        self.min_ch.blockSignals(False)
        self.max_ch.blockSignals(False)


class MUNFile:
    """
    MUN 3D TEM file object
    """

    def __init__(self):
        self.filepath = None

        self.data_type = None
        self.units = None
        self.ch_times = pd.Series()
        self.data = pd.DataFrame()

    def parse(self, filepath):
        self.filepath = Path(filepath)

        if not self.filepath.is_file():
            raise ValueError(f"{self.filepath} is not a file.")

        print(f"Parsing {self.filepath.name}")
        with open(filepath, 'r') as file:
            content = file.read()
            split_content = content.split('\n')

        # The top two lines of headers
        header = split_content[0].split('; ')
        self.data_type = header[0]
        self.units = re.sub('UNIT:', '', header[1]).strip()
        num_stations = int(split_content[1].split(': ')[1])
        stations = np.array(split_content[2].split(':')[1].split()).astype(float).astype(int)

        # Data
        data_match = [n.split() for n in split_content[3 + num_stations + 2: -1]]
        # First column is the channel number, second is the channel time, then it's the station numbers
        cols = ['Channel', 'ch_time']
        cols.extend(stations.astype(str))
        data = pd.DataFrame(data_match, columns=cols).transpose()
        self.ch_times = data.loc['ch_time']
        data.columns = data.loc['Channel']
        data.drop('ch_time', inplace=True)
        data.drop('Channel', inplace=True)

        # Make the station number a column instead of the index
        data = data.reset_index(level=0).rename({'index': 'Station'}, axis=1)
        data = data.astype(float)

        self.data = data
        print(f"Parsed data from {self.filepath.name}:\n{data}")
        return self


if __name__ == '__main__':
    parser = MUNFile()

    sample_files = Path(__file__).parents[2].joinpath('sample_files')
    file = sample_files.joinpath(r'MUN files\LONG_V1x1_450_50_100_50msec_3D_solution_channels_tem_time_decay_z.dat')
    mun_file = parser.parse(file)
