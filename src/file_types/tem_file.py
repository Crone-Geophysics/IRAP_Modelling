import re
from pathlib import Path

import matplotlib
from natsort import natsorted
import pandas as pd
import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (QLabel, QFormLayout, QWidget, QCheckBox, QFrame, QHBoxLayout, QSpinBox, QSizePolicy)
from matplotlib.pyplot import cm


class TEMTab(QWidget):
    toggle_sig = QtCore.pyqtSignal()

    def __init__(self, parent=None, color=None, axes=None):
        super().__init__()
        self.layout = QFormLayout()
        self.setLayout(self.layout)

        self.plot_cbox = QCheckBox("Plot")
        self.plot_cbox.setFocusPolicy(QtCore.Qt.NoFocus)
        self.plot_cbox.setChecked(True)

        self.layout.addRow(self.plot_cbox)
        self.layout.addRow("File Type", QLabel("Maxwell TEM"))

        # Channel selection frame
        self.ch_select_frame = QFrame()
        self.ch_select_frame.setLayout(QHBoxLayout())
        self.ch_select_frame.layout().setContentsMargins(0, 0, 0, 0)
        self.ch_select_frame.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.min_ch = QSpinBox()
        self.max_ch = QSpinBox()
        self.min_ch.setMinimum(1)
        self.max_ch.setMinimum(1)
        self.ch_select_frame.layout().addWidget(QLabel("Plot Channels"))
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
        self.min_ch.valueChanged.connect(lambda: self.update_channels("min"))
        self.max_ch.valueChanged.connect(lambda: self.update_channels("max"))

    def read(self, filepath):
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        ext = filepath.suffix.lower()

        if ext == '.tem':
            parser = TEMFile()
            try:
                file = parser.parse(filepath)
            except Exception as e:
                raise Exception(f"The following error occurred trying to parse the file: {e}.")
        else:
            raise ValueError(f"{ext} is not yet supported.")

        if file is None:
            raise ValueError(F"No data found in {filepath.name}.")

        # Add the file name as the default for the name in the legend
        # self.legend_name.setText(filepath.name)
        self.layout.addRow('Line', QLabel(file.line))
        self.layout.addRow('Configuration', QLabel(file.config))
        self.layout.addRow('Elevation', QLabel(str(file.elevation)))
        self.layout.addRow('Units', QLabel(file.units))
        self.layout.addRow('Current', QLabel(str(file.current)))

        self.layout.addRow('Rx Dipole', QLabel(str(file.rx_dipole)))

        if 'X' in file.components:
            self.layout.addRow('Rx Area X', QLabel(str(file.rx_area_x)))
        if 'Y' in file.components:
            self.layout.addRow('Rx Area Y', QLabel(str(file.rx_area_y)))
        if 'Z' in file.components:
            self.layout.addRow('Rx Area Z', QLabel(str(file.rx_area_z)))

        self.layout.addRow('Tx Dipole', QLabel(str(file.tx_dipole)))
        if file.tx_dipole:
            self.layout.addRow('Tx Moment', QLabel(str(file.tx_moment)))
        else:
            self.layout.addRow('Tx Turns', QLabel(str(file.tx_turns)))

        if file.components:
            self.layout.addRow('Components', QLabel('\n'.join(natsorted(file.components))))

        self.layout.addRow(self.ch_select_frame)
        # Create a data frame with channel times and channel widths
        channel_times = pd.DataFrame(zip(file.ch_times, file.ch_widths),
                                     columns=['Times', 'Widths'])
        channel_times.index += 1
        self.layout.addRow('Channel Times', QLabel(channel_times.to_string()))

        # Set the channel range spin boxes
        self.min_ch.blockSignals(True)
        self.max_ch.blockSignals(True)
        self.min_ch.setValue(1)
        self.max_ch.setValue(len(channel_times))
        self.min_ch.setMaximum(len(channel_times))
        self.max_ch.setMaximum(len(channel_times))
        self.min_ch.blockSignals(False)
        self.max_ch.blockSignals(False)

        if not file.loop_coords.empty:
            self.layout.addRow('Loop Coordinates', QLabel(file.loop_coords.to_string()))

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

        self.x_artists = []
        self.y_artists = []
        self.z_artists = []

        channels = [f'CH{num}' for num in range(1, len(self.file.ch_times) + 1)]
        plotting_channels = channels[self.min_ch.value() - 1: self.max_ch.value()]

        for component in self.file.components:
            comp_data = self.data[self.data.COMPONENT == component]

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
                    label = f"{ch}"
                # If coloring by line, uses the tab's color, and the label is the file name.
                else:
                    c = self.color
                    if ind == 0:
                        label = f"{self.file.filepath.name}"
                    else:
                        label = None

                x = comp_data.STATION.astype(float)
                y = comp_data.loc[:, ch].astype(float)

                if len(x) == 1:
                    style = 'o'
                    artist = ax.scatter(x, y,
                                        color=c,
                                        marker=style,
                                        s=size,
                                        alpha=alpha,
                                        label=label)

                else:
                    # style = '--' if 'Q' in freq else '-'
                    artist, = ax.plot(x, y,
                                      color=c,
                                      alpha=alpha,
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

        self.min_ch.blockSignals(False)
        self.max_ch.blockSignals(False)


class TEMFile:
    """
    Maxwell TEM file object
    """

    def __init__(self):
        self.filepath = None

        self.line = ''
        self.config = ''
        self.elevation = ''
        self.units = ''
        self.current = None
        self.components = []
        self.tx_turns = None
        self.base_freq = None
        self.duty_cycle = None
        self.on_time = None
        self.off_time = None
        self.turn_on = None
        self.turn_off = None
        self.timing_mark = None
        self.rx_area_x = None
        self.rx_area_y = None
        self.rx_area_z = None
        self.rx_dipole = False
        self.tx_dipole = False
        self.tx_moment = None  # Not sure if needed
        self.loop_coords = pd.DataFrame(columns=['Easting', 'Northing', 'Elevation'], dtype=float)
        self.ch_times = []
        self.ch_widths = []
        self.data = pd.DataFrame()

    def parse(self, filepath):
        self.filepath = Path(filepath)

        if not self.filepath.is_file():
            raise ValueError(f"{self.filepath} is not a file.")

        print(f"Parsing {self.filepath.name}")
        with open(filepath, 'r') as file:
            content = file.read()
            split_content = re.sub(' &', '', content).split('\n')

        # The top two lines of headers
        header = split_content[1].split()
        header.extend(split_content[2].split())
        header_dict = {}
        for match in header:
            value = match.split(':')
            header_dict[value[0]] = value[1]

        self.rx_dipole = True if header_dict['RXDIPOLE'] == 'YES' else False
        self.tx_dipole = True if header_dict['TXDIPOLE'] == 'YES' else False

        if not self.tx_dipole:
            # Parse the loop coordinates
            loop_coords_match = [c for c in split_content if 'LV' in c.upper()]
            loop_coords = []
            for match in loop_coords_match:
                if 'LV' in match:
                    values = [re.search(r'LV\d+\w:(.*)', m).group(1) for m in match.strip().split(' ')]
                    loop_coords.append(values)
            loop_coords = pd.DataFrame(loop_coords, columns=['Easting', 'Northing', 'Elevation']).astype(float)
            loop_coords.index += 1

            self.loop_coords = loop_coords

        # Channel times and widths
        ch_times = content.split(r'/TIMES(')[1].split('\n')[0][4:].split(',')
        ch_widths = content.split(r'/TIMESWIDTH(')[1].split('\n')[0][4:].split(',')

        # Data
        top_section, data_section = content.split(r'/PROFILEX:')
        data_columns = top_section.split('\n')[-2].split()
        data_match = data_section.split('\n')[1:]
        data = pd.DataFrame([match.split() for match in data_match[:-1]], columns=data_columns)
        data.iloc[:, 0:3] = data.iloc[:, 0:3].astype(float)
        data.iloc[:, 3] = data.iloc[:, 3].astype(float).astype(int)
        data.iloc[:, 4] = data.iloc[:, 4].astype(str)
        data.iloc[:, 5:] = data.iloc[:, 5:].astype(float)

        # Set the attributes
        self.line = header_dict['LINE']
        self.config = header_dict['CONFIG']
        self.elevation = header_dict['ELEV']
        self.units = re.sub('[()]', '', header_dict['UNITS'])
        self.current = header_dict['CURRENT']
        self.tx_turns = header_dict['TXTURNS']
        self.base_freq = header_dict['BFREQ']
        self.duty_cycle = header_dict['DUTYCYCLE']
        self.on_time = header_dict['ONTIME']
        self.off_time = header_dict['OFFTIME']
        self.turn_on = header_dict['TURNON']
        self.turn_off = header_dict['TURNOFF']
        self.timing_mark = header_dict['TIMINGMARK']
        self.rx_area_x = header_dict['RXAREAX']
        self.rx_area_y = header_dict['RXAREAY']
        self.rx_area_z = header_dict['RXAREAZ']
        self.ch_times = ch_times
        self.ch_widths = ch_widths
        self.data = data
        self.components = list(self.data.COMPONENT.unique())
        print(f"Parsed data from {self.filepath.name}:\n{data}")
        return self


if __name__ == '__main__':
    tem = TEMFile()

    sample_files = Path(__file__).parents[2].joinpath('sample_files')
    file = sample_files.joinpath(r'Maxwell files\V_1x1_450_50_100 50msec instant on-time first.tem')
    tem_file = tem.parse(file)
