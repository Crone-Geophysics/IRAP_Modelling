import pandas as pd
import numpy as np
from pathlib import Path


class MUNFile:
    """
    MUN 3D TEM file object
    """

    def __init__(self):
        self.filepath = None

        self.data_type = None
        self.units = None
        self.component = None
        self.ch_times = np.array([])
        self.data = pd.DataFrame()

    def parse(self, filepath):
        self.filepath = Path(filepath)

        if not self.filepath.is_file():
            raise ValueError(f"{self.filepath.name} is not a file.")

        print(f"Parsing {self.filepath.name}")
        with open(filepath, 'r') as file:
            content = file.read()
            split_content = content.split('\n')

        # The top two lines of headers
        header = split_content[0].split('; ')
        self.data_type = header[0]
        self.units = header[1]
        num_stations = int(split_content[1].split(': ')[1])
        stations = np.array(split_content[2].split(':')[1].split()).astype(float).astype(int)

        # Data
        data_match = [n.split() for n in split_content[3 + num_stations + 2: -1]]
        # First column is the channel number, second is the channel time, then it's the station numbers
        cols = ['Channel', 'ch_time']
        cols.extend(stations.astype(str))
        data = pd.DataFrame(data_match, columns=cols).transpose()
        self.ch_times = data.loc['ch_time'].to_numpy()
        data.columns = data.loc['Channel']
        data.drop('ch_time', inplace=True)
        data.drop('Channel', inplace=True)

        # Make the station number a column instead of the index
        data = data.reset_index(level=0).rename({'index': 'Station'}, axis=1)
        data = data.astype(float)

        self.data = data
        print(f"Parsed data from {self.filepath.name}:\n{data}")
        return self


if __name__ == '__main__':
    parser = MUNFile()

    sample_files = Path(__file__).parents[1].joinpath('sample_files')
    file = sample_files.joinpath(r'MUN files\LONG_V1x1_450_50_100_50msec_3D_solution_channels_tem_time_decay_z.dat')
    mun_file = parser.parse(file)
