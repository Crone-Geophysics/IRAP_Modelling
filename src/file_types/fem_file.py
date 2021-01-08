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

        self.plot_cbox = QCheckBox("Plot")
        self.plot_cbox.setFocusPolicy(QtCore.Qt.NoFocus)
        self.plot_cbox.setChecked(True)
        self.plot_cbox.toggled.connect(lambda: self.show_sig.emit())

        self.layout.addRow(self.plot_cbox)
        self.layout.addRow("File Type", QLabel("Maxwell FEM"))

        # self.legend_name = QLineEdit()
        # self.layout.addRow("Legend Name", self.legend_name)

        self.file = None
        self.lines = []
        self.data = pd.DataFrame()

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
        # self.legend_name.setText(filepath.name)
        self.layout.addRow('Line', QLabel(file.line))
        self.layout.addRow('Configuration', QLabel(file.config))
        self.layout.addRow('Elevation', QLabel(str(file.elevation)))
        self.layout.addRow('Units', QLabel(file.units))
        self.layout.addRow('Current', QLabel(str(file.current)))

        self.layout.addRow('Rx Dipole', QLabel(str(file.rx_dipole)))

        if file.rx_dipole:
            self.layout.addRow('Rx Area (HCP)', QLabel(file.rx_area_hcp))
        else:
            self.layout.addRow('Rx Area X', QLabel(file.rx_area_x))
            self.layout.addRow('Rx Area Y', QLabel(file.rx_area_y))
            self.layout.addRow('Rx Area Z', QLabel(file.rx_area_z))

        self.layout.addRow('Tx Dipole', QLabel(str(file.tx_dipole)))
        if file.tx_dipole:
            self.layout.addRow('Tx Moment', QLabel(file.tx_moment))
        else:
            self.layout.addRow('Tx Turns', QLabel(file.tx_turns))

        if file.rx_dipole and file.tx_dipole:
            self.layout.addRow('Horizontal Separation', QLabel(file.h_sep))
            self.layout.addRow('Vertical Separation', QLabel(file.v_sep))

        self.layout.addRow('Frequencies', QLabel(', '.join(file.frequencies)))

        if not file.loop_coords.empty:
            self.layout.addRow('Loop Coordinates', QLabel(file.loop_coords.to_string()))
        self.data = file.data
        self.file = file

    def plot(self, axes, color, alpha):
        """
        Plot the data on a mpl axes
        """
        self.lines = []
        count = 10

        for freq in self.file.frequencies:
            x = self.data.STATION.astype(float)
            y = self.data.loc[:, freq].astype(float)

            if len(x) == 1:
                style = 'x' if 'Q' in freq else 'o'
                line = axes.scatter(x, y,
                                    color=color,
                                    marker=style,
                                    s=count,
                                    alpha=alpha,
                                    label=f"{freq} ({self.file.filepath.name})")

            else:
                style = '--' if 'Q' in freq else '-'
                line, = axes.plot(x, y,
                                  ls=style,
                                  color=color,
                                  alpha=alpha,
                                  label=f"{freq} ({self.file.filepath.name})")
            self.lines.append(line)
            count += 10

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
