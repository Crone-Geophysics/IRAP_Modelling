import re
from pathlib import Path

import pandas as pd
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QLabel, QLineEdit, QFormLayout, QWidget, QCheckBox)


class FEMTab(QWidget):
    show_sig = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.layout = QFormLayout()
        self.setLayout(self.layout)

        self.plot_cbox = QCheckBox("Show")
        self.plot_cbox.setChecked(True)
        self.plot_cbox.toggled.connect(lambda: self.show_sig.emit())

        self.legend_name = QLineEdit()

        self.layout.addRow(self.plot_cbox)
        self.layout.addRow("File Type", QLabel("Maxwell FEM"))
        self.layout.addRow("Legend Name", self.legend_name)

        self.f = None
        self.lines = []
        self.data = pd.DataFrame()

    def read(self, filepath):
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        ext = filepath.suffix.lower()

        if ext == '.fem':
            parser = FEMFile()
            try:
                f = parser.parse(filepath)
            except Exception as e:
                raise Exception(f"The following error occurred trying to parse the file: {e}.")
        else:
            raise ValueError(f"{ext} is not yet supported.")

        if f is None:
            raise ValueError(F"No data found in {filepath.name}.")

        # Add the file name as the default for the name in the legend
        self.legend_name.setText(f.line)
        self.layout.addRow('Configuration', QLabel(f.config))
        self.layout.addRow('Elevation', QLabel(str(f.elevation)))
        self.layout.addRow('Units', QLabel(f.units))
        self.layout.addRow('Current', QLabel(str(f.current)))

        self.layout.addRow('Rx Dipole', QLabel(str(f.rx_dipole)))

        if f.rx_dipole:
            self.layout.addRow('Rx Area (HCP)', QLabel(f.rx_area_hcp))
        else:
            self.layout.addRow('Rx Area X', QLabel(f.rx_area_x))
            self.layout.addRow('Rx Area Y', QLabel(f.rx_area_y))
            self.layout.addRow('Rx Area Z', QLabel(f.rx_area_z))

        self.layout.addRow('Tx Dipole', QLabel(str(f.tx_dipole)))
        if f.tx_dipole:
            self.layout.addRow('Tx Moment', QLabel(f.tx_moment))
        else:
            self.layout.addRow('Tx Turns', QLabel(f.tx_turns))

        if f.rx_dipole and f.tx_dipole:
            self.layout.addRow('Horizontal Separation', QLabel(f.h_sep))
            self.layout.addRow('Vertical Separation', QLabel(f.v_sep))

        self.layout.addRow('Frequencies', QLabel(', '.join(f.frequencies)))

        if not f.loop_coords.empty:
            self.layout.addRow('Loop Coordinates', QLabel(f.loop_coords.to_string()))
        self.data = f.data
        self.f = f

    def plot(self, axes):
        """
        Plot the data on a mpl axes
        """
        self.lines = []

        for freq in self.f.frequencies:
            style = '--' if 'Q' in freq else '-'
            line, = axes.plot(self.data.STATION.astype(float), self.data.loc[:, freq].astype(float),
                              ls=style,
                              label=f"{freq} ({self.f.filepath.name})")
            self.lines.append(line)

        return self.lines


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

        self.tx_moment = None
        self.tx_turns = None
        self.h_sep = None
        self.v_sep = None
        self.rx_area_hcp = None
        self.rx_area_x = None
        self.rx_area_y = None
        self.rx_area_z = None
        self.rx_dipole = False
        self.tx_dipole = False

        self.frequencies = []
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

            self.loop_coords = loop_coords

        frequencies = content.split(r'/FREQ=')[1].split('\n')[0].split(',')
        assert frequencies, f"No frequencies found."

        # Data
        top_section, data_section = content.split(r'/PROFILEX:')
        data_columns = top_section.split('\n')[-2].split()
        data_match = data_section.split('\n')[1:]

        # Create the data frame
        data = pd.DataFrame([match.split() for match in data_match[:-1]], columns=data_columns)
        # data = data.astype(float)

        # Set the attributes
        self.line = header_dict['LINE']
        self.config = header_dict['CONFIG']
        self.elevation = header_dict['ELEV']
        self.units = re.sub('[()]', '', header_dict['UNITS'])
        self.current = header_dict['CURRENT']

        if self.rx_dipole:
            self.rx_area_hcp = header_dict['RXAREAHCP']
        else:
            # TODO See if there's RX Area when RX is not dipole
            self.rx_area_x = header_dict['RXAREAX']
            self.rx_area_y = header_dict['RXAREAY']
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
        self.data = data
        print(f"Parsed data from {self.filepath.name}:\n{data}")
        return self


if __name__ == '__main__':
    fem = FEMFile()

    sample_files = Path(__file__).parents[2].joinpath('sample_files')
    # file = sample_files.joinpath(r'Maxwell files\Test #2.fem')
    file = sample_files.joinpath(r'Maxwell files\Test 4 FEM files\Test 4 - h=5m.fem')
    fem_file = fem.parse(file)
