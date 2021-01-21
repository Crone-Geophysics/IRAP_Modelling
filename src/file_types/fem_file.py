import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QLabel, QFormLayout, QWidget, QCheckBox, QDoubleSpinBox, QSizePolicy, QSpinBox)
from natsort import natsorted


class FEMTab(QWidget):
    plot_changed_sig = QtCore.pyqtSignal()

    def __init__(self, parent=None, color=None, axes=None):
        super().__init__()

        self.layout = QFormLayout()
        self.setLayout(self.layout)

        self.plot_cbox = QCheckBox("Plot")
        self.plot_cbox.setFocusPolicy(QtCore.Qt.NoFocus)
        self.plot_cbox.setChecked(True)

        self.layout.addRow(self.plot_cbox)
        self.layout.addRow("File Type", QLabel("Maxwell FEM"))

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

        self.file = None
        self.parent = parent
        self.color = color
        self.alpha = None  # Remember the last alpha
        self.axes = axes

        self.hcp_artists = []
        self.vca_artists = []
        self.data = pd.DataFrame()

        # Signals
        self.plot_cbox.toggled.connect(self.toggle)
        self.scale_data_sbox.valueChanged.connect(self.plot)
        self.scale_data_sbox.valueChanged.connect(lambda: self.plot_changed_sig.emit())
        self.shift_stations_sbox.valueChanged.connect(self.plot)
        self.shift_stations_sbox.valueChanged.connect(lambda: self.plot_changed_sig.emit())
        self.alpha_sbox.valueChanged.connect(lambda: self.plot(color_by_channel=self.color_by_channel))
        self.alpha_sbox.valueChanged.connect(lambda: self.plot_changed_sig.emit())

    def read(self, filepath):
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        ext = filepath.suffix.lower()

        if ext == '.fem':
            parser = FEMFile()
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
        self.layout.addRow('Current', QLabel(str(file.current)))

        self.layout.addRow('Rx Dipole', QLabel(str(file.rx_dipole)))

        if file.rx_area_hcp:
            self.layout.addRow('Rx Area (HCP)', QLabel(file.rx_area_hcp))
        if file.rx_area_vca:
            self.layout.addRow('Rx Area (VCA)', QLabel(file.rx_area_vca))

        self.layout.addRow('Tx Dipole', QLabel(str(file.tx_dipole)))
        if file.tx_dipole:
            self.layout.addRow('Tx Moment', QLabel(str(file.tx_moment)))
        else:
            self.layout.addRow('Tx Turns', QLabel(str(file.tx_turns)))

        if file.rx_dipole and file.tx_dipole:
            self.layout.addRow('Horizontal Separation', QLabel(str(file.h_sep)))
            self.layout.addRow('Vertical Separation', QLabel(str(file.v_sep)))

        if file.frequencies:
            self.layout.addRow('Frequencies', QLabel('\n'.join(natsorted(file.frequencies))))

        if file.components:
            self.layout.addRow('Components', QLabel('\n'.join(natsorted(np.unique(file.components)))))

        if not file.loop_coords.empty:
            self.layout.addRow('Loop Coordinates', QLabel(file.loop_coords.to_string()))

        self.data = file.data
        self.file = file

    def plot(self):
        """
        Plot the data on a mpl axes
        :param alpha : float
        """
        # Remove existing plotted lines
        self.remove()

        self.hcp_artists = []
        self.vca_artists = []
        size = 10

        # When there's component-by-frequency
        if self.file.comp_by_freq:

            for freq, component in self.file.comp_by_freq.items():
                print(f"Plotting frequency {freq}")
                comp_data = self.data.loc[:, ['STATION', freq]]

                if comp_data.empty:
                    print(f"No data for component {component} and frequency {freq}.")
                    continue

                ax = self.axes[component]
                x = comp_data.STATION.astype(float) + self.shift_stations_sbox.value()
                y = comp_data.loc[:, freq].astype(float) * self.scale_data_sbox.value()

                if len(x) == 1:
                    style = 'x' if 'Q' in freq else 'o'
                    artist = ax.scatter(x, y,
                                        color=self.color,
                                        marker=style,
                                        s=size,
                                        alpha=self.alpha_sbox.value() / 100,
                                        label=f"{freq} ({self.file.filepath.name})")

                else:
                    style = '--' if 'Q' in freq else '-'
                    artist, = ax.plot(x, y,
                                      ls=style,
                                      color=self.color,
                                      alpha=self.alpha_sbox.value() / 100,
                                      label=f"{freq} ({self.file.filepath.name})")

                if component == 'HCP':
                    self.hcp_artists.append(artist)
                else:
                    self.vca_artists.append(artist)

                size += 10  # For scatter point size

        else:
            for freq in self.file.frequencies:
                print(f"Plotting {freq} frequency.")

                for component in self.file.components:
                    print(f"Plotting component {component}.")
                    ax = self.axes[component]

                    comp_data = self.data[self.data.COMPONENT == component]

                    if comp_data.empty:
                        print(f"No data for component {component} and frequency {freq}.")
                        continue

                    x = comp_data.STATION.astype(float)
                    y = comp_data.loc[:, freq].astype(float)

                    if len(x) == 1:
                        style = 'x' if 'Q' in freq else 'o'
                        artist = ax.scatter(x, y,
                                            color=self.color,
                                            marker=style,
                                            s=size,
                                            alpha=alpha,
                                            label=f"{freq} ({self.file.filepath.name})")

                    else:
                        style = '--' if 'Q' in freq else '-'
                        artist, = ax.plot(x, y,
                                          ls=style,
                                          color=self.color,
                                          alpha=alpha,
                                          label=f"{freq} ({self.file.filepath.name})")

                    if component == 'HCP':
                        self.hcp_artists.append(artist)
                    else:
                        self.vca_artists.append(artist)

                    size += 10  # For scatter point size

    def remove(self):
        # Remove existing plotted lines
        for ls, ax in zip([self.hcp_artists, self.vca_artists], self.axes.values()):
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
            if ax == self.axes['HCP']:
                artists = self.hcp_artists
            else:
                artists = self.vca_artists

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


class FEMFile:
    """
    Maxwell FEM file object
    """
    def __init__(self):
        self.filepath = None

        self.line = None
        self.config = None
        self.elevation = 0.
        self.units = '%Ht'
        self.current = 1.
        self.components = []

        self.tx_moment = None
        self.tx_turns = None
        self.h_sep = None
        self.v_sep = None
        self.rx_area_hcp = None
        self.rx_area_vca = None
        self.rx_area_x = None
        self.rx_area_y = None
        self.rx_area_z = None
        self.rx_dipole = False
        self.tx_dipole = False

        self.frequencies = []
        self.comp_by_freq = {}
        self.loop_coords = pd.DataFrame()
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
        # Ignore loop name because the spaces in the name causes problems with files that are dipole tx
        if 'loop' not in split_content[2].lower():
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

        frequencies = content.split(r'/FREQ=')[1].split('\n')[0].split(',')
        assert frequencies, f"No frequencies found."

        # Data
        top_section, data_section = content.split(r'/PROFILEX:')
        data_columns = top_section.split('\n')[-2].split()
        data_match = data_section.split('\n')[1:]

        # Create the data frame
        data = pd.DataFrame([match.split() for match in data_match[:-1]], columns=data_columns)

        # Set the attributes
        self.line = header_dict['LINE']
        self.config = header_dict['CONFIG']
        self.elevation = header_dict['ELEV']
        self.units = re.sub('[()]', '', header_dict['UNITS'])
        self.current = header_dict['CURRENT']

        if self.rx_dipole:
            if 'RXAREAHCP' in header_dict.keys():
                self.rx_area_hcp = header_dict['RXAREAHCP']
            if 'RXAREAVCA' in header_dict.keys():
                self.rx_area_vca = header_dict['RXAREAVCA']
        else:
            # TODO See if there's RX Area when RX is not dipole
            if 'RXAREAX' in header_dict.keys():
                self.rx_area_x = header_dict['RXAREAX']
            if 'RXAREAY' in header_dict.keys():
                self.rx_area_y = header_dict['RXAREAY']
            if 'RXAREAZ' in header_dict.keys():
                self.rx_area_z = header_dict['RXAREAZ']

        if self.tx_dipole:
            self.tx_moment = header_dict['TXMOMENT']
        else:
            self.tx_turns = header_dict['TXTURNS']

        # TODO make sure this is true, that sep and vsep only exist when both rx and tx are dipoles
        if self.tx_dipole and self.rx_dipole:
            self.h_sep = header_dict['SEP']
            self.v_sep = header_dict['VSEP']

        self.frequencies = frequencies

        if 'COMPONENT' not in data.columns:
            components = content.split(r'/COMPONENTBYFREQ=')[1].split('\n')[0].split(',')
            if len(components) != len(frequencies):
                raise ValueError(F"The number of frequencies does not match the components by frequencies.")

            # If all components are the same
            if all([components[0] == comp for comp in components]):
                component = components[0]
                data['COMPONENT'] = component
                self.components = [component]
            else:
                self.comp_by_freq = dict(zip(frequencies, components))
                self.components = components
                data['COMPONENT'] = ', '.join(components)
        else:
            self.components = list(data.COMPONENT.unique())
        self.data = data

        print(f"Parsed data from {self.filepath.name}:\n{data}")
        return self


if __name__ == '__main__':
    fem = FEMFile()

    sample_files = Path(__file__).parents[2].joinpath('sample_files')
    # file = sample_files.joinpath(r'Maxwell files\Test #2.fem')
    file = sample_files.joinpath(r'Maxwell files\Test 4 FEM files\Test 4 - h=5m.fem')
    fem_file = fem.parse(file)
