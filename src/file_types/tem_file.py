import re
from pathlib import Path

import pandas as pd
from PyQt5.QtWidgets import (QLabel)
from natsort import natsorted

from src.file_types.base_tdem_file import BaseTDEM


class TEMTab(BaseTDEM):

    def __init__(self, parent=None, axes=None):
        super().__init__(parent=parent, axes=axes)
        self.layout.insertRow(1, "File Type", QLabel("Maxwell TEM"))
        self.color = 'b'

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
        self.layout.addRow('Line', QLabel(file.line))
        self.layout.addRow('Configuration', QLabel(file.config))
        self.layout.addRow('Elevation', QLabel(str(file.elevation)))
        self.layout.addRow('Units', QLabel(file.units))
        self.layout.addRow('Current', QLabel(f"{float(file.current):,}"))

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

        self.layout.addRow(QLabel("Plot Channels"), self.ch_select_frame)
        # Create a data frame with channel times and channel widths
        channel_times = pd.DataFrame(zip(file.ch_times, file.ch_widths),
                                     columns=['Times', 'Widths'])
        channel_times.index += 1
        self.layout.addRow('Channel Times', QLabel(channel_times.to_string()))

        # Set the channel range spin boxes
        self.min_ch.blockSignals(True)
        self.max_ch.blockSignals(True)
        self.min_ch.setValue(1)
        self.min_ch.setMaximum(len(channel_times))
        self.max_ch.setMaximum(len(channel_times))
        self.max_ch.setValue(len(channel_times))
        self.min_ch.blockSignals(False)
        self.max_ch.blockSignals(False)

        if not file.loop_coords.empty:
            self.layout.addRow('Loop Coordinates', QLabel(file.loop_coords.to_string()))

        self.data = file.data
        self.file = file
        self.legend_name.setText(f"{self.file.filepath.stem} (Maxwell)")

    def plot(self, color_by_channel=None):
        """
        Plot the data on a mpl axes
        :param color_by_channel: bool, color each channel a different color or color each line with self.color.
        """
        # Remove existing plotted lines
        self.clear()

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
            ax = self.axes[component]

            for ind, ch in enumerate(plotting_channels):
                # If coloring by channel, uses the rainbow color iterator and the label is the channel number.
                if ind == 0:
                    label = self.legend_name.text()
                else:
                    label = None

                x = comp_data.STATION.astype(float) + self.shift_stations_sbox.value()
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

    def plot_decay(self):
        """
        Plot the decay data on a mpl axes
        """
        # Remove existing plotted lines
        self.clear()

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

            ax = self.axes[component]

            for ind, ch in enumerate(plotting_channels):
                # If coloring by channel, uses the rainbow color iterator and the label is the channel number.
                if ind == 0:
                    label = f"{self.file.filepath.stem} (Maxwell)"
                else:
                    label = None

                x = comp_data.STATION.astype(float)
                y = comp_data.loc[:, ch].astype(float)

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
        if 'RXAREAX' in header_dict.keys():
            self.rx_area_x = header_dict['RXAREAX']
        if 'RXAREAY' in header_dict.keys():
            self.rx_area_y = header_dict['RXAREAY']
        if 'RXAREAZ' in header_dict.keys():
            self.rx_area_z = header_dict['RXAREAZ']
        self.ch_times = ch_times
        self.ch_widths = ch_widths
        self.data = data
        self.components = list(self.data.COMPONENT.unique())
        # print(f"Parsed data from {self.filepath.name}:\n{data}")
        return self


if __name__ == '__main__':
    tem = TEMFile()

    sample_files = Path(__file__).parents[2].joinpath('sample_files')
    file = sample_files.joinpath(r'Maxwell files\V_1x1_450_50_100 50msec instant on-time first.tem')
    tem_file = tem.parse(file)
