from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QLabel)
from natsort import natsorted

from src.file_types.base_tdem_widget import BaseTDEM


class PlateFTab(BaseTDEM):

    def __init__(self, parent=None, axes=None):
        super().__init__(parent=parent, axes=axes)
        self.layout.insertRow(1, "File Type", QLabel("PlateF File"))

        self.color = "r"

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
        self.layout.addRow('Current', QLabel(f"{float(file.current):,}"))

        self.layout.addRow('Rx Area', QLabel(str(file.rx_area)))

        if file.components:
            self.layout.addRow('Components', QLabel('\n'.join(natsorted(file.components))))

        self.layout.addRow(QLabel("Plot Channels"), self.ch_select_frame)

        channel_times = file.ch_times.astype(float) * 1000
        channel_times.index += 1
        self.layout.addRow('Channel Times', QLabel(channel_times.to_string()))

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
        self.legend_name.setText(f"{self.file.filepath.stem} (PLATE)")

    def plot(self):
        """
        Plot the data on a mpl axes
        """
        # Remove existing plotted lines
        self.clear()

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

            ax = self.axes[component]

            for ind, ch in enumerate(plotting_channels):
                if ind == 0:
                    label = self.legend_name.text()
                else:
                    label = None

                x = comp_data.Station.astype(float) + self.shift_stations_sbox.value()
                y = comp_data.loc[:, ch].astype(float) * self.scale_data_sbox.value()

                if len(x) == 1:
                    style = 'o'
                    artist = ax.scatter(x, y,
                                        color=self.color,
                                        marker=style,
                                        s=size,
                                        alpha=self.alpha_sbox.value() / 100,
                                        label=label)

                else:
                    # style = '--' if 'Q' in freq else '-'
                    artist, = ax.plot(x, y,
                                      color=self.color,
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

        self.plot_changed_sig.emit()


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

        # print(f"Parsed data from {self.filepath.name}:\n{data}")
        return self


if __name__ == '__main__':
    platef = PlateFFile()

    sample_files = Path(__file__).parents[2].joinpath('sample_files')
    file = sample_files.joinpath(r'PLATEF files\450_50.dat')
    parsed_file = platef.parse(file)
