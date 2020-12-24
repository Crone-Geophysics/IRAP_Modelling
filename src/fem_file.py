import pandas as pd
import re
from pathlib import Path


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
        self.loop = None
        self.loop_coords = pd.DataFrame()
        self.data = pd.DataFrame()

    def parse(self, filepath):
        self.filepath = Path(filepath)

        if not self.filepath.is_file():
            raise ValueError(f"{self.filepath.name} is not a file.")

        with open(filepath, 'r') as file:
            content = file.read()
            split_content = re.sub(' &', '', content).split('\n')

        # The top two lines of headers
        header = split_content[1].split()
        header.extend([split_content[2]])
        header_dict = {}
        for match in header:
            value = match.split(':')
            header_dict[value[0]] = value[1]

        # Loop name
        loop = split_content[3][5:-2]  # Remove "LOOP:" and " &"

        # Parse the loop coordinates
        loop_coords_match = [c for c in split_content if 'LV' in c.upper()]
        loop_coords = []
        for match in loop_coords_match:
            if 'LV' in match:
                values = [re.search(r'LV\d+\w:(.*)', m).group(1) for m in match.strip().split(' ')]
                loop_coords.append(values)
        loop_coords = pd.DataFrame(loop_coords, columns=['Easting', 'Northing', 'Elevation']).astype(float)

        # TODO Need more frequencies
        global frequencies, cols
        frequencies = content.split(r'/FREQ=')[1].split('\n')[0].split(',')
        assert frequencies, f"No frequencies found."

        # Data
        data_match = content.split(r'/PROFILEX:')[1].split('\n')[1:]
        # Headers that are always there (?)
        cols = ['Easting', 'Northing', 'Elevation', 'Station', 'Component', 'Dircosz', 'Dircose', 'Dircosn']
        cols.extend(frequencies)
        cols.extend(['Distance', 'Calc_this'])
        data = pd.DataFrame([match.split() for match in data_match[:-1]], columns=cols)
        data.iloc[:, 0:4] = data.iloc[:, 0:4].astype(float)
        data.iloc[:, 5:] = data.iloc[:, 5:].astype(float)

        # Set the attributes
        self.line = header_dict['LINE']
        self.config = header_dict['CONFIG']
        self.elevation = header_dict['ELEV']
        self.units = header_dict['ELEV']
        self.current = header_dict['CURRENT']
        self.rx_dipole = True if header_dict['RXDIPOLE'] == 'YES' else False
        self.tx_dipole = True if header_dict['TXDIPOLE'] == 'YES' else False

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
        self.loop = loop
        self.loop_coords = loop_coords
        self.data = pd.DataFrame()

        return self


if __name__ == '__main__':
    fem = FEMFile()

    # file = r'C:\Users\Mortulo\PycharmProjects\IRAP_Modelling\sample_files\Maxwell files\Test #2.fem'
    file = r'C:\Users\kajth\PycharmProjects\IRAP_Modelling\sample_files\Maxwell files\Test 4 FEM files\Test 4 - h=5m.fem'
    fem_file = fem.parse(file)
