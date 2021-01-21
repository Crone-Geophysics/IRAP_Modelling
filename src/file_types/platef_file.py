import re
from pathlib import Path

import matplotlib
from natsort import natsorted
import pandas as pd
import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (QLabel, QFormLayout, QWidget, QCheckBox, QFrame, QHBoxLayout, QSpinBox, QDoubleSpinBox,
                             QSizePolicy)
from matplotlib.pyplot import cm


class PlateFTab(QWidget):
    plot_changed_sig = QtCore.pyqtSignal()

    def __init__(self, parent=None, color=None, axes=None):
        super().__init__()
        self.layout = QFormLayout()
        self.setLayout(self.layout)

        self.plot_cbox = QCheckBox("Plot")
        self.plot_cbox.setFocusPolicy(QtCore.Qt.NoFocus)
        self.plot_cbox.setChecked(True)

        self.layout.addRow(self.plot_cbox)
        self.layout.addRow("File Type", QLabel("PlateF File"))

        # Data editing
        self.scale_data_sbox = QDoubleSpinBox()
        self.scale_data_sbox.setValue(1.)
        self.scale_data_sbox.setSingleStep(0.1)
        self.scale_data_sbox.setMaximum(1e9)
        self.scale_data_sbox.setMinimum(-1e9)
        self.scale_data_sbox.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.layout.addRow("Scale Data", self.scale_data_sbox)

        self.shift_stations_sbox = QDoubleSpinBox()
        self.shift_stations_sbox.setMaximum(100000)
        self.shift_stations_sbox.setMinimum(-100000)
        self.shift_stations_sbox.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.layout.addRow("Shift Stations", self.shift_stations_sbox)

        self.alpha_sbox = QSpinBox()
        self.alpha_sbox.setSingleStep(10)
        self.alpha_sbox.setRange(0, 100)
        self.alpha_sbox.setValue(100)
        self.alpha_sbox.setSuffix('%')
        self.alpha_sbox.setFixedWidth(100)
        self.layout.addRow("Plot Alpha", self.alpha_sbox)

        # Channel selection frame
        self.ch_select_frame = QFrame()
        self.ch_select_frame.setLayout(QHBoxLayout())
        self.ch_select_frame.layout().setContentsMargins(0, 0, 0, 0)
        self.ch_select_frame.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.min_ch = QSpinBox()
        self.max_ch = QSpinBox()
        self.min_ch.setMinimum(1)
        self.max_ch.setMinimum(1)
        self.min_ch.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.max_ch.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        # self.ch_select_frame.layout().addWidget(QLabel("Plot Channels"))
        self.ch_select_frame.layout().addWidget(self.min_ch)
        self.ch_select_frame.layout().addWidget(QLabel("to"))
        self.ch_select_frame.layout().addWidget(self.max_ch)

        self.file = None
        self.parent = parent
        self.color = color
        self.alpha = None  # Remember the last alpha
        self.color_by_channel = False
        self.axes = axes

        self.x_artists = []
        self.y_artists = []
        self.z_artists = []
        self.data = pd.DataFrame()

        # Signals
        self.plot_cbox.toggled.connect(self.toggle)
        self.scale_data_sbox.valueChanged.connect(lambda: self.plot(color_by_channel=self.color_by_channel))
        self.scale_data_sbox.valueChanged.connect(lambda: self.plot_changed_sig.emit())
        self.shift_stations_sbox.valueChanged.connect(lambda: self.plot(color_by_channel=self.color_by_channel))
        self.shift_stations_sbox.valueChanged.connect(lambda: self.plot_changed_sig.emit())
        self.alpha_sbox.valueChanged.connect(lambda: self.plot(color_by_channel=self.color_by_channel))
        self.alpha_sbox.valueChanged.connect(lambda: self.plot_changed_sig.emit())
        self.min_ch.valueChanged.connect(lambda: self.update_channels("min"))
        self.max_ch.valueChanged.connect(lambda: self.update_channels("max"))

    def read(self, filepath):
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        ext = filepath.suffix.lower()

        if ext == '.dat':
            parser = PlateFFile()
            try:
                file = parser.parse(filepath)
            except Exception as e:
                raise Exception(f"The following error occurred trying to parse the file: {e}.")
        else:
            raise ValueError(f"{ext} is not yet supported.")

        if file is None:
            raise ValueError(F"No data found in {filepath.name}.")

        # Add the file name as the default for the name in the legend
        # self.layout.addRow('Units', QLabel(file.units))
        self.layout.addRow('Current', QLabel(str(file.current)))

        self.layout.addRow('Rx Area', QLabel(str(file.rx_area)))

        if file.components:
            self.layout.addRow('Components', QLabel('\n'.join(natsorted(file.components))))

        self.layout.addRow(QLabel("Plot Channels"), self.ch_select_frame)
        self.layout.addRow('Channel Times', QLabel((file.ch_times.astype(float) * 1000).to_string()))

        # Set the channel range spin boxes
        self.min_ch.blockSignals(True)
        self.max_ch.blockSignals(True)
        self.min_ch.setValue(1)
        self.min_ch.setMaximum(len(file.ch_times))
        self.max_ch.setMaximum(len(file.ch_times))
        self.max_ch.setValue(len(file.ch_times))
        self.min_ch.blockSignals(False)
        self.max_ch.blockSignals(False)

        # if not file.loop_coords.empty:
        #     self.layout.addRow('Loop Coordinates', QLabel(file.loop_coords.to_string()))

        self.data = file.data
        self.file = file

    def plot(self, color_by_channel=None):
        """
        Plot the data on a mpl axes
        :param alpha: float
        :param color_by_channel: bool, color each channel a different color or color each line with self.color.
        """
        # Remove existing plotted lines
        self.remove()

        # Use the current legend coloring if none is passed
        if color_by_channel is None:
            color_by_channel = self.color_by_channel
        else:
            self.color_by_channel = color_by_channel

        self.x_artists = []
        self.y_artists = []
        self.z_artists = []

        channels = [f'{num}' for num in range(1, len(self.file.ch_times) + 1)]
        plotting_channels = channels[self.min_ch.value() - 1: self.max_ch.value()]

        for component in self.file.components:
            comp_data = self.data[self.data.Component == component]

            if comp_data.empty:
                print(f"No {component} data in {self.file.filepath.name}.")
                continue

            size = 8  # For scatter point size

            if color_by_channel is True:
                rainbow_color = iter(cm.gist_rainbow(np.linspace(0, 1, len(plotting_channels))))

            ax = self.axes[component]

            for ind, ch in enumerate(plotting_channels):
                # If coloring by channel, uses the rainbow color iterator and the label is the channel number.
                if color_by_channel is True:
                    c = next(rainbow_color)  # Cycles through colors
                    ch_num = int(re.search(r'\d+', ch).group(0)) - 1
                    label = f"{ch} ({self.file.ch_times[ch_num]} ms)"
                # If coloring by line, uses the tab's color, and the label is the file name.
                else:
                    c = self.color
                    if ind == 0:
                        label = f"{self.file.filepath.name}"
                    else:
                        label = None

                x = comp_data.Station.astype(float) + self.shift_stations_sbox.value()
                y = comp_data.loc[:, ch].astype(float) * self.scale_data_sbox.value()

                if len(x) == 1:
                    style = 'o'
                    artist = ax.scatter(x, y,
                                        color=c,
                                        marker=style,
                                        s=size,
                                        alpha=self.alpha_sbox.value() / 100,
                                        label=label)

                else:
                    # style = '--' if 'Q' in freq else '-'
                    artist, = ax.plot(x, y,
                                      color=c,
                                      alpha=self.alpha_sbox.value() / 100,
                                      # lw=count / 100,
                                      label=label)

                if component == 'X':
                    self.x_artists.append(artist)
                elif component == 'Y':
                    self.y_artists.append(artist)
                else:
                    self.z_artists.append(artist)

                size += 2

    def remove(self):
        # Remove existing plotted lines
        for ls, ax in zip([self.x_artists, self.y_artists, self.z_artists], self.axes.values()):
            if ax.lines:
                if all([artist in ax.lines for artist in ls]):
                    for artist in ls:
                        ax.lines.remove(artist)
            if ax.collections:
                if all([artist in ax.collections for artist in ls]):
                    for artist in ls:
                        ax.collections.remove(artist)

    def toggle(self):
        """Toggle the visibility of plotted lines/points"""
        for ax in self.axes.values():
            if ax == self.axes['X']:
                artists = self.x_artists
            elif ax == self.axes['Y']:
                artists = self.y_artists
            else:
                artists = self.z_artists

            lines = list(filter(lambda x: isinstance(x, matplotlib.lines.Line2D), artists))
            points = list(filter(lambda x: isinstance(x, matplotlib.collections.PathCollection), artists))  # Scatters

            if lines:
                if self.plot_cbox.isChecked():
                    if all([a in ax.lines for a in lines]):  # If the lines are already plotted, pass.
                        pass
                    else:
                        for artist in artists:
                            ax.lines.append(artist)
                else:
                    for artist in artists:
                        ax.lines.remove(artist)

            if points:
                # Add or remove the scatter points
                if self.plot_cbox.isChecked():
                    if all([a in ax.collections for a in points]):  # If the points are already plotted, pass.
                        pass
                    else:
                        for artist in artists:
                            ax.collections.append(artist)
                else:
                    for artist in artists:
                        ax.collections.remove(artist)

        self.plot_changed_sig.emit()

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
        self.plot_changed_sig.emit()  # Updates the legend and re-draws

        self.min_ch.blockSignals(False)
        self.max_ch.blockSignals(False)

        for ax in self.axes.values():
            # Re-scale the plots
            ax.relim()
            ax.autoscale()


class PlateFFile:
    """
    PLATEF TEM file object
    """

    def __init__(self):
        self.filepath = None

        self.ch_times = pd.Series()
        self.current = None
        self.rx_area = None
        self.units = 'nT/s'
        self.data = pd.DataFrame()
        self.components = []

    def parse(self, filepath):
        self.filepath = Path(filepath)

        if not self.filepath.is_file():
            raise ValueError(f"{self.filepath} is not a file.")

        print(f"Parsing {self.filepath.name}")
        with open(filepath, 'r') as file:
            content = file.read()
            split_content = content.split('\n')

        # The top two lines of headers
        data_start = int(split_content[1].split()[0])
        # loop_coords = [s.split()[:3] for s in split_content[8:12]]
        ch_times_match = split_content[data_start-13:data_start-1]
        ch_times = np.array(' '.join(ch_times_match).split(), dtype=float)
        ch_times = np.ma.masked_equal(ch_times, 0).compressed()  # Remove all 0s
        num_channels = len(ch_times)  # Number of off-time channels

        current = float(split_content[data_start-15].split()[1])
        rx_area = float(split_content[data_start-15].split()[2])

        # Data
        data_match = np.array(' '.join(split_content[data_start:]).split())
        data_match = np.reshape(data_match, (int(len(data_match) / (num_channels + 3)), num_channels + 3))

        # Create a data frame
        cols = ['Station', 'Component', '0']
        cols.extend(np.arange(1, num_channels + 1).astype(str))
        data = pd.DataFrame(data_match, columns=cols)
        data.iloc[:, 0] = data.iloc[:, 0].astype(float).astype(int)
        data.iloc[:, 2:] = data.iloc[:, 2:].astype(float)

        # Set the attributes
        self.data = data
        self.ch_times = pd.Series(ch_times)
        self.current = current
        self.rx_area = rx_area
        self.components = list(self.data.Component.unique())

        print(f"Parsed data from {self.filepath.name}:\n{data}")
        return self


if __name__ == '__main__':
    platef = PlateFFile()

    sample_files = Path(__file__).parents[2].joinpath('sample_files')
    file = sample_files.joinpath(r'PLATEF files\450_50.dat')
    parsed_file = platef.parse(file)
