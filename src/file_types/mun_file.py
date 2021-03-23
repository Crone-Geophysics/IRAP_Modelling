import re
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QLabel)

from src.file_types.base_tdem_widget import BaseTDEM
from src.post_process_by_JL import read_em3d_raw, read_observation_line, \
    plot_decay_curve, read_tem_file, write_time_decay_files, \
    plot_multi_channel


class MUNTab(BaseTDEM):

    def __init__(self, parent=None, axes=None, component=None):
        super().__init__(parent=parent, axes=axes)
        self.layout.insertRow(1, "File Type", QLabel("MUN File"))

        self.component = component
        self.color = "g"

    def read(self, filepath):
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        ext = filepath.suffix.lower()

        if ext == '.dat':
            parser = MUNFile()
            try:
                file = parser.parse(filepath)
            except Exception as e:
                raise Exception(f"The following error occurred trying to parse the file: {e}.")
        else:
            raise ValueError(f"{ext} is not yet supported.")

        if file is None:
            raise ValueError(F"No data found in {filepath.name}.")

        # Add the file name as the default for the name in the legend
        self.layout.addRow('Units', QLabel(file.units))
        self.layout.addRow('Component', QLabel(self.component))

        self.layout.addRow(QLabel("Plot Channels"), self.ch_select_frame)
        # Create a data frame with channel times and channel widths
        file.ch_times.name = 'Times'
        file.ch_times.index += 1
        self.layout.addRow('Channel Times', QLabel(file.ch_times.to_string()))

        # Set the channel range spin boxes
        self.min_ch.blockSignals(True)
        self.max_ch.blockSignals(True)
        self.min_ch.setValue(1)
        self.min_ch.setMaximum(len(file.ch_times))
        self.max_ch.setMaximum(len(file.ch_times))
        self.max_ch.setValue(len(file.ch_times))
        self.min_ch.blockSignals(False)
        self.max_ch.blockSignals(False)

        self.data = file.data
        self.file = file
        self.legend_name.setText(f"{self.file.filepath.stem} (MUN)")

    def plot(self):
        """
        Plot the data on a mpl axes
        """
        # Remove existing plotted lines
        self.clear()

        channels = [f'{num}' for num in range(1, len(self.file.ch_times) + 1)]
        plotting_channels = channels[self.min_ch.value() - 1: self.max_ch.value()]

        data = self.data

        if data.empty:
            print(f"No {self.component} data in {self.file.filepath.name}.")
            return

        size = 8  # For scatter point size

        for ind, ch in enumerate(plotting_channels):
            if ind == 0:
                label = self.legend_name.text()
            else:
                label = None

            x = data.Station.astype(float) + self.shift_stations_sbox.value()
            y = data.loc[:, ch].astype(float) * self.scale_data_sbox.value()

            if len(x) == 1:
                style = 'o'
                artist = self.axes[self.component].scatter(x, y,
                                                           color=self.color,
                                                           marker=style,
                                                           s=size,
                                                           alpha=self.alpha_sbox.value() / 100,
                                                           label=label)

            else:
                # style = '--' if 'Q' in freq else '-'
                artist, = self.axes[self.component].plot(x, y,
                                                         color=self.color,
                                                         alpha=self.alpha_sbox.value() / 100,
                                                         # lw=count / 100,
                                                         label=label)

            if self.component == 'X':
                self.x_artists.append(artist)
            elif self.component == 'Y':
                self.y_artists.append(artist)
            else:
                self.z_artists.append(artist)

            size += 2

        self.plot_changed_sig.emit()


class MUNFile:
    """
    MUN 3D TEM file object
    """

    def __init__(self):
        self.filepath = None

        self.data_type = None
        self.units = None
        self.ch_times = pd.Series()
        self.data = pd.DataFrame()

    @staticmethod
    def convert(filepath):
        """
        Create a txt file for each model inside Peter's text file. Saves the files in the same directory.
        :param filepath: Path or str
        """

        def convert_files_in_folder(folder_dir):
            print(f"Plotting data files in {str(folder_dir)}.")

            # primary_directory = Path(str(folder_dir) + r"_primary")
            # if not primary_directory.exists():
            #     print(f"Could not find the primary directory for {str(folder_dir)}.")
            #     return
            # primary_data = read_data(primary_directory)

            def read_data(data_folder_dir):
                """Read the data from the multiple files inside Jianbo's raw results folder"""
                data_file = list(data_folder_dir.glob("*_dBdt.DAT"))
                channels_file = list(data_folder_dir.glob("time_iterations.DAT"))
                stations_file = list(data_folder_dir.glob("*observation_points_coordinates.XYZ"))

                if not data_file:
                    print(f"Could not find the dB/dt file in {str(data_folder_dir)}.")
                    return
                if not channels_file:
                    print(f"Could not find the channels file in {str(data_folder_dir)}.")
                    return
                if not stations_file:
                    print(f"Could not find the stations file in {str(data_folder_dir)}.")
                    return

                data_file = data_file[0]
                channels_file = channels_file[0]
                stations_file = stations_file[0]

                maxwell_channels = np.array([-1.9500, -1.8500, -1.7500, -1.6500, -1.5500, -1.4495, -1.3500,
                                             -1.2500, -1.1500, -1.0500, -0.9500, -0.8500, -0.7500, -0.6500,
                                             -0.5500, -0.4500, -0.3500, -0.2500, -0.1500, -0.0500, 0.0255,
                                             0.0550, 0.0700, 0.0950, 0.1300, 0.1750, 0.2350, 0.3150, 0.4200,
                                             0.5600, 0.7450, 0.9900, 1.3150, 1.7450, 2.3150, 3.0750, 4.0850,
                                             5.4250, 7.2050, 9.5700, 12.6600, 22.7300, 39.3950, 48.1150, 48.5255,
                                             48.5550, 48.5700, 48.5950, 48.6300, 48.6750, 48.7350, 48.8150, 48.9200,
                                             49.0600, 49.2450, 49.4900, 49.8150, 50.2450, 50.8150, 51.5750, 52.5850,
                                             53.9250, 55.7050, 58.0700, 61.1600, 71.2300, 87.8950, 97.1150]) + 2.
                # # convert Maxwell channels to seconds
                # maxwell_channels = maxwell_channels * 1e-3

                # Data read is in units of T
                raw_data = pd.read_csv(str(data_file), delim_whitespace=True, header=None) * 1e9
                raw_data.columns = ["X", "Y", "Z"]
                raw_channels = pd.read_csv(str(channels_file), delim_whitespace=True, header=None).to_numpy() * 1e3
                stations = pd.read_csv(str(stations_file), delim_whitespace=True, header=None)
                stations.columns = ["Station", "Easting", "Northing", "Elevation"]

                kept_channels = []
                for ch in maxwell_channels:
                    # Find the nearest channel index in channels
                    ind = min(range(len(raw_channels)), key=lambda i: abs(raw_channels[i]-ch))
                    print(f"Nearest channel for value of {ch:.3f}: {ind}")
                    kept_channels.append(ind)

                raw_channels = pd.DataFrame(raw_channels)
                channel_filt = raw_channels.index.isin(kept_channels)
                channels = raw_channels[channel_filt].to_numpy()
                # Build data frame
                data = pd.DataFrame()
                for i, station in stations.iterrows():
                    x_data = raw_data.iloc[kept_channels, 0].reset_index(drop=True)
                    y_data = raw_data.iloc[kept_channels, 1].reset_index(drop=True)
                    z_data = raw_data.iloc[kept_channels, 2].reset_index(drop=True)
                    station_data = pd.DataFrame([x_data, y_data, z_data]).reset_index(drop=False)
                    station_data = station_data.rename({"index": "Component"}, axis=1)

                    # station_data = raw_data.iloc[kept_channels]
                    station_data.insert(0, "Station", int(station.Station))
                    station_data.insert(1, "Easting", int(station.Easting))
                    station_data.insert(2, "Northing", int(station.Northing))
                    station_data.insert(3, "Elevation", int(station.Elevation))
                    data = data.append(station_data)

                data.reset_index(inplace=True, drop=True)
                return data, channels

            data, channels = read_data(folder_dir)






            decay_or_profile = 1   # 1: decaying plots;  2: profile plots
            
            if decay_or_profile == 1:
                decaying_plot = True
                profile_plot = False
            elif decay_or_profile == 2:
                decaying_plot = False
                profile_plot = True

            if decaying_plot is True:
                channel_start = 1
                channel_end = 68
                site = 201
                outputFileName_basic = "Basetime_50ms_plate_100S_"+asp_ratio+"_dBdt_"  # for time decay plot
                extra_mark = "_semiLogScale_1Period"
                #extra_mark = "_semiLogScale_QuarterPeriod"   # extra marking string to add for output file name;
                write_decay_file = False  # whether to write time decay file onto disk
            else:
                channel_start = 21
                channel_end = 44
                # extra_mark_multi_list = ["_OFF1", "_OFF2", "_OFF3", "_OFF4"]
                # channel_start_list = [1, 9, 15, 21] # OFF1 takes the chs of 1-8 of total off-time chs, OFF2 takes 9-14
                #                                     # , etc
                # channel_end_list = [8, 14, 20, 24]
                extra_mark_multi_list = ["_OFF1", "_OFF2", "_OFF3", "_OFF4", "_OFF5"]
                channel_start_list = [1, 9, 15, 19, 21] # OFF1 takes the chs of 1-8 of total off-time chs, OFF2 takes 9-14
                # , etc
                channel_end_list = [8, 14, 18, 20, 22]
                outputFileName_multi = "Basetime_50ms_plate_100S_"+asp_ratio+"_dBdt_profile_"  # for multi-channel plot
            
            # channels_Maxwell = discrete_chs[channel_start - 1 : channel_end]
            #print(channels_Maxwell)
            
            normal_or_not = False   # whether normalize results, true or false
            plate_cond = "plate=100 S"   # conductivity string marker
            x_is_log = False
            y_is_log = True
            unit_magnetic = "nT"
            baseTime = 50.e-3  # msec
            rampTime = 1.5e-3  # msec, linear ramp off
            
            # evenly distributed time instants(channels) in seconds; for 3D code
            if y_is_log is True:
                # data amount for plotting
                nt_data = 500
            else:
                nt_data = 1000
            # if decaying_plot is True:
            channels_3D = np.linspace(-6.14e-3, 148.4e-3, nt_data, dtype=np.float)   # 50 ms, 1 periods
            # if profile_plot is True:
            #     # when plotting multi-channel
            #     channels_3D = channels_Maxwell
            
            # First we need to read in the field values
            field_filename = str(folder_dir) + 'iTr=001_dBdt.dat'
            print("--3D data file: ", field_filename)

            # This is the filename of the time-stepping schemes
            timeStep_filename = str(folder_dir) + 'time_stepping_scheme.txt'
            
            # The number of observation sites
            obs_filename = str(folder_dir) + 'iTr=001_observation_points_coordinates.xyz'
            stn = read_observation_line(obs_filename, whichColumn=2)
            n_rec = stn.size
            
            # The number of actual time iterations
            time_iters_filename = str(folder_dir) + 'time_iterations.dat'
            time_iters = np.loadtxt(time_iters_filename, dtype=np.float)
            n_iters = time_iters.size
            
            # Read in the 3D modeled response. field stores EM fields in the format:f(n_channel, n_location, n_component)
            # 3D data unit: T/s
            field_3D = read_em3d_raw(field_filename, n_rec, n_iters, channels_3D, timeStep_filename,
                                     ZeroTimeShift=None, interp=False)
            # read the primary 3D response
            field_pri_filename = primary_directory + 'iTr=001_dBdt.dat'
            timeStep_filename = primary_directory + 'time_stepping_scheme.txt'
            time_iters_filename = primary_directory + 'time_iterations.dat'
            time_iters = np.loadtxt(time_iters_filename, dtype=np.float)
            n_iters = time_iters.size
            
            # when asking primary fields over more than 1 period
            nch_total = channels_3D.size
            nch_redueced_1T = np.zeros(nch_total) # shifted time channels
            
            period = baseTime * 4.0
            for k in range(nch_total):
                nch_redueced_1T[k] = (channels_3D[k]  + baseTime + rampTime)   # since t=-51.5 ms is the beginning of everything
                ratio = nch_redueced_1T[k] / period
                nch_redueced_1T[k] = nch_redueced_1T[k] - np.floor(ratio) * period
            
            nch_redueced_1T = nch_redueced_1T - (baseTime + rampTime)
            field_pri_3D = read_em3d_raw(field_pri_filename, n_rec, n_iters, nch_redueced_1T, timeStep_filename,
                                         ZeroTimeShift=None, interp=False)
            
            field_3D = field_3D - field_pri_3D
            
            if unit_magnetic == "nT":
                field_3D = field_3D * 1e+9
            if decaying_plot is True:
                if write_decay_file is True:
                    print("--write data files in the directory:", PDF_directory)
                    write_time_decay_files(channels_3D, stn, np.transpose(field_3D[:,:,0]), np.transpose(field_3D[:,:,1]),
                                           np.transpose(field_3D[:,:,2]),path=PDF_directory,
                                           title="LONG_V1x1_450_50_1000_16msec_3D_All_channels")




        filepath = Path(filepath)
        if not filepath.is_dir():
            raise ValueError(f"{str(filepath)} is not a directory.")

        print(f"Converting files in {filepath.name}")

        all_files = list(filepath.rglob(r"*.txt"))
        data_folders = [f.parent for f in all_files if "primary" not in str(f)]

        for folder in data_folders:
            convert_files_in_folder(folder)

    def parse(self, filepath):
        self.filepath = Path(filepath)

        if not self.filepath.is_file():
            raise ValueError(f"{self.filepath} is not a file.")

        print(f"Parsing {self.filepath.name}")
        with open(filepath, 'r') as file:
            content = file.read()
            split_content = content.split('\n')

        # The top two lines of headers
        header = split_content[0].split('; ')
        self.data_type = header[0]
        self.units = re.sub('UNIT:', '', header[1]).strip()
        num_stations = int(split_content[1].split(': ')[1])
        stations = np.array(split_content[2].split(':')[1].split()).astype(float).astype(int)

        # Data
        data_match = [n.split() for n in split_content[3 + num_stations + 2: -1]]
        # First column is the channel number, second is the channel time, then it's the station numbers
        cols = ['Channel', 'ch_time']
        cols.extend(stations.astype(str))
        data = pd.DataFrame(data_match, columns=cols).transpose()
        self.ch_times = data.loc['ch_time']
        data.columns = data.loc['Channel']
        data.drop('ch_time', inplace=True)
        data.drop('Channel', inplace=True)

        # Make the station number a column instead of the index
        data = data.reset_index(level=0).rename({'index': 'Station'}, axis=1)
        data = data.astype(float)

        self.data = data
        # print(f"Parsed data from {self.filepath.name}:\n{data}")
        return self


if __name__ == '__main__':
    parser = MUNFile()

    # sample_files = Path(__file__).parents[2].joinpath('sample_files')
    # file = sample_files.joinpath(r'MUN files\LONG_V1x1_450_50_100_50msec_3D_solution_channels_tem_time_decay_z.dat')
    # mun_file = parser.parse(file)

    folder = r"A:\IRAP\All_3D_data_files\Aspect Ratio\150m"
    mun_file = MUNFile()
    mun_file.convert(folder)
