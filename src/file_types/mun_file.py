import re
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QLabel)

from src.file_types.base_tdem_widget import BaseTDEM


class MUNTab(BaseTDEM):

    def __init__(self, parent=None, axes=None, component=None):
        super().__init__(parent=parent, axes=axes)
        self.layout.insertRow(1, "File Type", QLabel("MUN File"))

        self.component = component
        self.color = "g"

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

        self.layout.addRow(QLabel("Plot Channels"), self.ch_select_frame)
        # Create a data frame with channel times and channel widths
        file.ch_times.name = 'Times'
        file.ch_times.index += 1
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
        self.legend_name.setText(f"{self.file.filepath.stem} (MUN)")

    def plot(self):
        """
        Plot the data on a mpl axes
        """
        # Remove existing plotted lines
        self.clear()

        channels = [f'{num}' for num in range(1, len(self.file.ch_times) + 1)]
        plotting_channels = channels[self.min_ch.value() - 1: self.max_ch.value()]

        data = self.data

        if data.empty:
            print(f"No {self.component} data in {self.file.filepath.name}.")
            return

        size = 8  # For scatter point size

        for ind, ch in enumerate(plotting_channels):
            if ind == 0:
                label = self.legend_name.text()
            else:
                label = None

            x = data.Station.astype(float) + self.shift_stations_sbox.value()
            y = data.loc[:, ch].astype(float) * self.scale_data_sbox.value()

            if len(x) == 1:
                style = 'o'
                artist = self.axes[self.component].scatter(x, y,
                                                           color=self.color,
                                                           marker=style,
                                                           s=size,
                                                           alpha=self.alpha_sbox.value() / 100,
                                                           label=label)

            else:
                # style = '--' if 'Q' in freq else '-'
                artist, = self.axes[self.component].plot(x, y,
                                                         color=self.color,
                                                         alpha=self.alpha_sbox.value() / 100,
                                                         # lw=count / 100,
                                                         label=label)

            if self.component == 'X':
                self.x_artists.append(artist)
            elif self.component == 'Y':
                self.y_artists.append(artist)
            else:
                self.z_artists.append(artist)

            size += 2

        self.plot_changed_sig.emit()


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
        # print(f"Parsed data from {self.filepath.name}:\n{data}")
        return self


if __name__ == '__main__':
    parser = MUNFile()

    sample_files = Path(__file__).parents[2].joinpath('sample_files')
    file = sample_files.joinpath(r'MUN files\LONG_V1x1_450_50_100_50msec_3D_solution_channels_tem_time_decay_z.dat')
    mun_file = parser.parse(file)
