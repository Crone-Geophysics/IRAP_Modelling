import re
from pathlib import Path

from natsort import natsorted
import pandas as pd
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QLabel, QFormLayout, QWidget, QCheckBox)
from matplotlib.pyplot import cm


class TEMTab(QWidget):
    show_sig = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()

        self.layout = QFormLayout()
        self.setLayout(self.layout)

        self.plot_cbox = QCheckBox("Plot")
        self.plot_cbox.setFocusPolicy(QtCore.Qt.NoFocus)
        self.plot_cbox.setChecked(True)
        self.plot_cbox.toggled.connect(lambda: self.show_sig.emit())

        self.layout.addRow(self.plot_cbox)
        self.layout.addRow("File Type", QLabel("Maxwell TEM"))

        # self.legend_name = QLineEdit()
        # self.layout.addRow("Legend Name", self.legend_name)

        self.file = None
        self.x_artists = []
        self.y_artists = []
        self.z_artists = []
        self.data = pd.DataFrame()

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

        # Create a data frame with channel times and channel widths
        channel_times = pd.DataFrame(zip(file.ch_times, file.ch_widths),
                                     columns=['Times', 'Widths'])
        self.layout.addRow('Channel Times', QLabel(channel_times.to_string()))

        if not file.loop_coords.empty:
            self.layout.addRow('Loop Coordinates', QLabel(file.loop_coords.to_string()))

        self.data = file.data
        self.file = file

    def plot(self, axes, color, alpha):
        """
        Plot the data on a mpl axes
        :param axes: dict of axes object for each component.
        :param color: str, the color of all lines/scatter points.
        :param alpha : float
        """
        self.x_artists = []
        self.y_artists = []
        self.z_artists = []

        count = 10

        for component in self.file.components:
            rainbow_colors = iter(cm.gist_rainbow(np.linspace(0, 1, len(self.file.ch_times))))
            comp_data = self.data[self.data.COMPONENT == component]

            ax = axes[component]
            for ch in [f'CH{num}' for num in range(1, len(self.file.ch_times) + 1)]:
                c = next(rainbow_colors)  # Cycles through colors
                x = comp_data.STATION.astype(float)
                y = comp_data.loc[:, ch].astype(float)
                if len(x) == 1:
                    style = 'o'
                    artist = ax.scatter(x, y,
                                        color=c,
                                        marker=style,
                                        s=count,
                                        alpha=alpha,
                                        label=f"{ch} ({self.file.filepath.name})")

                else:
                    # style = '--' if 'Q' in freq else '-'
                    artist, = ax.plot(x, y,
                                      color=c,
                                      alpha=alpha,
                                      # lw=count / 1000,
                                      label=f"{ch} ({self.file.filepath.name})")

                if component == 'X':
                    self.x_artists.append(artist)
                elif component == 'Y':
                    self.y_artists.append(artist)
                else:
                    self.z_artists.append(artist)

                count += 10  # For scatter point size


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

            self.loop_coords = loop_coords

        global ch_times, cols
        # Channel times and widths
        ch_times = content.split(r'/TIMES(')[1].split('\n')[0][4:].split(',')
        ch_widths = content.split(r'/TIMESWIDTH(')[1].split('\n')[0][4:].split(',')

        # Data
        top_section, data_section = content.split(r'/PROFILEX:')
        data_columns = top_section.split('\n')[-2].split()
        data_match = data_section.split('\n')[1:]
        # data_match = content.split(r'/PROFILEX:')[1].split('\n')[1:]
        # Headers that are always there (?)
        # cols = ['Easting', 'Northing', 'Elevation', 'Station', 'Component', 'Dircosz', 'Dircose', 'Dircosn']
        # Add the channel numbers as column names
        # cols.extend(np.arange(0, len(ch_times)).astype(str))
        # cols.extend(['Distance', 'Calc_this'])
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
        self.components = self.data.COMPONENT.unique()
        print(f"Parsed data from {self.filepath.name}:\n{data}")
        return self


if __name__ == '__main__':
    tem = TEMFile()

    sample_files = Path(__file__).parents[2].joinpath('sample_files')
    file = sample_files.joinpath(r'Maxwell files\V_1x1_450_50_100 50msec instant on-time first.tem')
    tem_file = tem.parse(file)
