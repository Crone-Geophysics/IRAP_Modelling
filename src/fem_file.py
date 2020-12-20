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
        self.tx_moment = 1.
        self.h_sep = 1.
        self.v_sep = 0.
        self.rx_area_hcp = 1.
        self.rx_dipole = False
        self.tx_dipole = False
        self.frequencies = []
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
        header.extend(split_content[2].split())
        header_dict = {}
        for match in header:
            value = match.split(':')
            header_dict[value[0]] = value[1]

        # TODO Need more frequencies

        global frequencies, cols
        frequencies = content.split(r'/FREQ=')[1].split('\n')[0].split(',')

        # Data

        def data_to_readings(row):
            reading = row.iloc[len(cols):len(ch_times) - len(cols)].to_numpy().astype(float)
            return reading

        data_match = content.split(r'/PROFILEX:')[1].split('\n')[1:]
        # Headers that are always there (?)
        cols = ['Easting', 'Northing', 'Elevation', 'Station', 'Component', 'Dircosz', 'Dircose', 'Dircosn']
        data = pd.DataFrame([match.split() for match in data_match[:-1]])
        data.iloc[:, 0:4] = data.iloc[:, 0:4].astype(float)
        data.iloc[:, 5:] = data.iloc[:, 5:].astype(float)

        # Combine the channel column readings into numpy arrays
        readings = data.apply(data_to_readings, axis=1)

        # Remove the old channel columns
        data = data.drop(columns=range(len(cols), len(cols) + len(ch_times)))
        # Update the column names
        cols.extend(['Distance', 'Calc_this', 'Reading'])
        data['Reading'] = readings
        data.columns = cols

        # Set the attributes
        self.line = header_dict['LINE']
        self.config = header_dict['CONFIG']
        self.elevation = header_dict['ELEV']
        self.units = header_dict['ELEV']
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
        self.rx_dipole = header_dict['RXDIPOLE']
        self.tx_dipole = header_dict['TXDIPOLE']
        self.loop = loop
        self.loop_coords = loop_coords
        self.ch_times = ch_times
        self.ch_widths = ch_widths
        self.data = pd.DataFrame()

        return self


if __name__ == '__main__':
    fem = FEMFile()

    file = r'C:\Users\Mortulo\PycharmProjects\IRAP_Modelling\sample_files\Maxwell files\Test #2.fem'
    fem_file = fem.parse(file)
