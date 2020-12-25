import pandas as pd
import numpy as np
import re
from pathlib import Path


class TEMFile:
    """
    Maxwell TEM file object
    """

    def __init__(self):
        self.filepath = None

        self.line = None
        self.config = 'fixed_loop'
        self.elevation = 0.
        self.units = 'nT/s'
        self.current = 1.
        self.tx_turns = 1.
        self.base_freq = 1.
        self.duty_cycle = 50.
        self.on_time = 50.
        self.off_time = 50.
        self.turn_on = 0.
        self.turn_off = 0.
        self.timing_mark = 0.
        self.rx_area_x = 4000.
        self.rx_area_y = 4000.
        self.rx_area_z = 4000.
        self.rx_dipole = False
        self.tx_dipole = False
        self.loop = None
        self.loop_coords = pd.DataFrame(columns=['Easting', 'Northing', 'Elevation'], dtype=float)
        self.ch_times = []
        self.ch_widths = []
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

        # Loop name
        loop = split_content[3].split(':')[1]

        # Parse the loop coordinates
        loop_coords_match = [c for c in split_content if 'LV' in c.upper()]
        loop_coords = []
        for match in loop_coords_match:
            if 'LV' in match:
                values = [re.search(r'LV\d+\w:(.*)', m).group(1) for m in match.strip().split(' ')]
                loop_coords.append(values)
        loop_coords = pd.DataFrame(loop_coords, columns=['Easting', 'Northing', 'Elevation']).astype(float)

        global ch_times, cols
        # Channel times and widths
        ch_times = content.split(r'/TIMES(')[1].split('\n')[0][4:].split(',')
        ch_widths = content.split(r'/TIMESWIDTH(')[1].split('\n')[0][4:].split(',')

        # Data
        data_match = content.split(r'/PROFILEX:')[1].split('\n')[1:]
        # Headers that are always there (?)
        cols = ['Easting', 'Northing', 'Elevation', 'Station', 'Component', 'Dircosz', 'Dircose', 'Dircosn']
        # Add the channel numbers as column names
        cols.extend(np.arange(0, len(ch_times)).astype(str))
        cols.extend(['Distance', 'Calc_this'])
        data = pd.DataFrame([match.split() for match in data_match[:-1]], columns=cols)
        data.iloc[:, 0:3] = data.iloc[:, 0:3].astype(float)
        data.iloc[:, 3] = data.iloc[:, 3].astype(float).astype(int)
        data.iloc[:, 4] = data.iloc[:, 4].astype(str)
        data.iloc[:, 5:] = data.iloc[:, 5:].astype(float)

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
        self.data = data

        return self


if __name__ == '__main__':
    tem = TEMFile()

    sample_files = Path(__file__).parents[1].joinpath('sample_files')
    file = sample_files.joinpath(r'Maxwell files\V_1x1_450_50_100 50msec instant on-time first.tem')
    tem_file = tem.parse(file)
