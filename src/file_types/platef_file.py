import re
from pathlib import Path

import pandas as pd
from PyQt5.QtWidgets import (QLabel, QLineEdit, QFormLayout, QWidget, QCheckBox)


class PlateFTab(QWidget):

    def __init__(self):
        super().__init__()
        self.layout = QFormLayout()
        self.setLayout(self.layout)

        self.plot_cbox = QCheckBox("Show")
        self.legend_name = QLineEdit()

        self.layout.addRow(self.plot_cbox)
        self.layout.addRow("Legend Name", self.legend_name)

    def read(self, filepath):
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        ext = filepath.suffix.lower()

        if ext == '.dat':
            parser = PlateFFile()
            f = parser.parse(filepath)
        else:
            raise ValueError(f"{ext} is not yet supported.")

        if f is None:
            raise ValueError(F"No data found in {filepath.name}.")


class PlateFFile:
    """
    PLATEF TEM file object
    """

    def __init__(self):
        self.filepath = None

        self.ch_times = np.array([])
        self.current = None
        self.rx_area = None
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
        self.ch_times = ch_times
        self.current = current
        self.rx_area = rx_area

        print(f"Parsed data from {self.filepath.name}:\n{data}")
        return self


if __name__ == '__main__':
    platef = PlateFFile()

    sample_files = Path(__file__).parents[2].joinpath('sample_files')
    file = sample_files.joinpath(r'PLATEF files\450_50.dat')
    parsed_file = platef.parse(file)
