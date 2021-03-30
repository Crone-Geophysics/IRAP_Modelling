import re
from pathlib import Path

import numpy as np
import pandas as pd
import time
import os
import math
from PyQt5.QtWidgets import (QLabel)

from src.file_types.base_tdem_widget import BaseTDEM
from src.post_process_by_JL import read_em3d_raw, read_observation_line


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
        self.ch_times = pd.Series(dtype=float)
        self.data = pd.DataFrame()

    @staticmethod
    def convert(folder, primary_folder=None, output_folder=None, ar=False):
        """
        Convert Jianbo's data folder into TEM-like files.
        :param folder: str or Path, parent folder which contains model data. Subfolders inside this folder should have
        the data files.
        :param primary_folder: str or Path, parent folder which contains the primary field data.
        :param output_folder: str or Path, folder to save all the files to.
        :param ar: Bool, if converting Aspect Ratio test (for naming)
        """

        def get_field_data(data_folder, channels, primary_field=False):
            # read the primary 3D response
            field_filename = data_folder.joinpath(r'iTr=001_dBdt.dat')
            channel_file = data_folder.joinpath(r'time_stepping_scheme.txt')
            time_iters_filename = data_folder.joinpath(r'time_iterations.dat')
            obs_filename = data_folder.joinpath(r'iTr=001_observation_points_coordinates.xyz')
            assert field_filename.exists(), F"Cannot find dBdt data file for the primary field in {primary_folder}."
            assert channel_file.exists(), F"Cannot find channel times file in {primary_folder}."
            assert time_iters_filename.exists(), F"Cannot find time stepping iterations file in {primary_folder}."
            assert obs_filename.exists(), F"Cannot find observation sites file in {primary_folder}."

            # Number of stations
            stn = read_observation_line(str(obs_filename), whichColumn=2)
            n_rec = stn.size

            # The number of actual time iterations in the 3D model
            time_iters = np.loadtxt(str(time_iters_filename), dtype=float)
            n_iters = time_iters.size

            # Reduce the periods if it's the primary field folder
            if primary_field is True:
                # when asking primary fields over more than 1 period
                timebase = 50.e-3  # msec
                ramp = 1.5e-3  # msec, linear ramp off
                nch_total = channels.size
                nch_redueced_1t = np.zeros(nch_total)  # shifted time channels

                period = timebase * 4.0
                for k in range(nch_total):
                    nch_redueced_1t[k] = (channels[k] + timebase + ramp)  # t=-51.5 ms is the beginning of everything
                    ratio = nch_redueced_1t[k] / period
                    nch_redueced_1t[k] = nch_redueced_1t[k] - np.floor(ratio) * period

                channels = nch_redueced_1t - (timebase + ramp)

            # Read in the 3D modeled response. 3D data unit: T/s
            field_3d = read_em3d_raw(str(field_filename), n_rec, n_iters, channels, str(channel_file),
                                     ZeroTimeShift=None, interp=False)
            return field_3d

        def write_time_decay_files(channels, stations, fieldx, fieldy, fieldz, out_path):
            """
            Write the text file in a format similar to a TEM file.
            :param channels: list of floats, channel times.
            :param stations: list of floats, station numbers.
            :param fieldx: 2D np array
            :param fieldy: 2D np array
            :param fieldz: 2D np array
            :param out_path: str, filepath of the output text file.
            """
            channels = channels * 1e3
            with open(out_path, 'w') as file:

                num_stations, num_channels = fieldx.shape
                station_names = [f"{i + 1}" for i in stations]
                station_names = [f"{i:^8}" for i in station_names]
                channel_names = [f"CH{i + 1}" for i in range(len(channels))]
                channel_names = [f"{i:^15}" for i in channel_names]

                file.write("Data type: dB/dt; UNIT: nT/s\n")
                file.write(f"Number of stations: {num_stations}\n")
                file.write(f"Stations (m): {' '.join([str(s) for s in stations]):^10}\n")
                file.write(f"Channel times (ms):\n")

                # Add the channel times
                for i in range(len(channels)):
                    file.write(f"{i + 1:^8}{channels[i]:^ 8.4f}\n")

                file.write("EM data:\n")
                file.write(f"{'Station':^8}{'Component':^8}{''.join(channel_names)}\n")

                # Add the data
                print(f"Stations {stations.min()} - {stations.max()}")
                for i in range(num_stations):
                    x = [f"{x:^ 15.5E}" for x in fieldx[i, :]]
                    y = [f"{x:^ 15.5E}" for x in fieldy[i, :]]
                    z = [f"{x:^ 15.5E}" for x in fieldz[i, :]]
                    file.write(f"{station_names[i]:^8}{'X':^8}{''.join(x)}\n")
                    file.write(f"{station_names[i]:^8}{'Y':^8}{''.join(y)}\n")
                    file.write(f"{station_names[i]:^8}{'Z':^8}{''.join(z)}\n")

        folder = Path(folder)
        if primary_folder is None:
            primary_folder = list(folder.glob(r"*_primary"))
            assert primary_folder, f"{str(primary_folder)} is not a directory."
            primary_folder = primary_folder[0]
        else:
            primary_folder = Path(primary_folder)
            assert primary_folder, f"{str(primary_folder)} is not a directory."

        print(f"Converting files in {folder.name}")
        t = time.time()

        assert folder.is_dir(), f"{str(folder)} is not a directory."

        data_results = [p for p in list(folder.rglob(r"*")) if p.is_dir() and "primary" not in str(p)]
        assert data_results, F"No folders found in {folder.name}."

        channels = np.array([-1.9500, -1.8500, -1.7500, -1.6500, -1.5500, -1.4495, -1.3500,
                             -1.2500, -1.1500, -1.0500, -0.9500, -0.8500, -0.7500, -0.6500,
                             -0.5500, -0.4500, -0.3500, -0.2500, -0.1500, -0.0500, 0.0255,
                             0.0550, 0.0700, 0.0950, 0.1300, 0.1750, 0.2350, 0.3150, 0.4200,
                             0.5600, 0.7450, 0.9900, 1.3150, 1.7450, 2.3150, 3.0750, 4.0850,
                             5.4250, 7.2050, 9.5700, 12.6600, 22.7300, 39.3950, 48.1150, 48.5255,
                             48.5550, 48.5700, 48.5950, 48.6300, 48.6750, 48.7350, 48.8150, 48.9200,
                             49.0600, 49.2450, 49.4900, 49.8150, 50.2450, 50.8150, 51.5750, 52.5850,
                             53.9250, 55.7050, 58.0700, 61.1600, 71.2300, 87.8950, 97.1150])
        channels = channels * 1e-3

        field_pri_3d = get_field_data(primary_folder, channels, primary_field=True)

        for data_folder in data_results:
            print(f"Converting {data_folder}.")
            print(f"Using primary folder {primary_folder}")

            obs_filename = data_folder.joinpath(r'iTr=001_observation_points_coordinates.xyz')
            stn = read_observation_line(str(obs_filename), whichColumn=2)

            # Read in the 3D modeled response. 3D data unit: T/s
            field_3d = get_field_data(data_folder, channels, primary_field=False)

            assert len(stn) == field_3d.shape[1], f"Number of stations is not equal to size of field_3d ({len(stn)} vs {field_3d.shape[1]})"
            field_3d = field_3d - field_pri_3d
            field_3d = field_3d * 1e+9  # For nT

            if output_folder is None:
                output_folder = folder
            else:
                output_folder = Path(output_folder)

            if ar is True:
                """ Aspect Ratio naming """
                conductance = re.sub(r"results_50msec_", "", str(data_folder.name), flags=re.I)
                conductance = re.sub(r"_set1", "", conductance, flags=re.IGNORECASE).upper()
                conductance = re.sub(r"_set2", "", conductance, flags=re.IGNORECASE).upper()
                if conductance == "100S":
                    letter = "A"
                elif conductance == "1KS":
                    letter = "B"
                elif conductance == "10KS":
                    letter = "C"
                else:
                    raise ValueError(F"{conductance} is invalid.")
                """ Aspect Ratio naming END """

                out_path = output_folder.joinpath(data_folder.parent.name).with_suffix(".DAT")
                out_name = re.sub("m", letter, out_path.name)
                out_path = out_path.with_name(out_name)
            else:
                out_path = output_folder.joinpath(data_folder.parent.name).with_suffix(".DAT")

            write_time_decay_files(channels,
                                   stn,
                                   np.transpose(field_3d[:, :, 0]),
                                   np.transpose(field_3d[:, :, 1]),
                                   np.transpose(field_3d[:, :, 2]),
                                   str(out_path))

        print(F"Conversion process complete.")
        print(f"Conversion time for {folder.name}: {int(math.floor((time.time() - t) / 60)):02d}:{int(time.time() % 60):02d}.")

    def parse(self, filepath):
        self.filepath = Path(filepath)

        if not self.filepath.is_file():
            raise ValueError(f"{self.filepath} is not a file.")

        print(f"Parsing {self.filepath.name}")
        with open(filepath, 'r') as file:
            content = file.read()
            split_content = content.split('\n')

        header = split_content[0].split('; ')
        self.data_type = header[0]
        self.units = re.sub(r'UNIT:', '', header[1]).strip()
        channels = content.split(r"Channel times (ms):")[-1].split(r"EM data:")[0].strip()
        self.ch_times = pd.Series([c.split()[-1] for c in channels.split("\n")], dtype=float)

        # Data
        data_match = content.split("EM data:\n")[-1].split("\n")
        data_match = [d.split() for d in data_match]
        data = pd.DataFrame.from_records(data_match).dropna()
        # Use the first row as the column names, then remove the first row
        data.columns = data.iloc[0]
        data.drop(axis=0, index=0, inplace=True)
        data.iloc[:, 2:] = data.iloc[:, 2:].astype(float)
        self.data = data
        # print(f"Parsed data from {self.filepath.name}:\n{data}")
        return self


if __name__ == '__main__':
    parser = MUNFile()

    def convert_folders():
        t = time.time()

        # """Two-way induction"""
        # base_folder = r"A:\IRAP\All_3D_data_files_formatted\Two-way induction"
        # sub_folders = Path(base_folder).glob(r"*")
        # sub_folders = [s for s in sub_folders if s.is_dir() and "Model5" not in str(s)]
        # out_folder = samples_folder.joinpath(r"Two-way induction\300x100\100S\MUN")
        #
        # converter = MUNFile()
        # for ind, folder in enumerate(sub_folders):
        #     if "Model" in str(folder):
        #         continue
        #     print(f"Converting folder {ind + 1}/{len(sub_folders)}.")
        #     converter.convert(folder,
        #                       output_folder=out_folder,
        #                       ar=False)

        # """Aspect Ratio 150m"""
        # base_folder = r"A:\IRAP\All_3D_data_files_formatted\Aspect Ratio\150m"
        # sub_folders = Path(base_folder).glob(r"*")
        # sub_folders = [s for s in sub_folders if s.is_dir() and "plots" not in str(s)]
        # out_folder = samples_folder.joinpath(r"Aspect ratio\MUN")
        #
        # converter = MUNFile()
        # for ind, folder in enumerate(sub_folders):
        #     if "Model" in str(folder):
        #         continue
        #     print(f"Converting folder {ind + 1}/{len(sub_folders)}.")
        #     converter.convert(folder,
        #                       output_folder=out_folder,
        #                       ar=True)

        """Aspect Ratio 600m"""
        base_folder = r"A:\IRAP\All_3D_data_files_formatted\Aspect Ratio\600m"
        sub_folders = Path(base_folder).glob(r"*")
        sub_folders = [s for s in sub_folders if s.is_dir() and "plots" not in str(s)]
        out_folder = samples_folder.joinpath(r"Aspect ratio\MUN")

        converter = MUNFile()
        for ind, folder in enumerate(sub_folders):
            if "Model" in str(folder):
                continue
            print(f"Converting folder {ind + 1}/{len(sub_folders)}.")
            converter.convert(folder,
                              primary_folder=r"A:\IRAP\All_3D_data_files_formatted\Aspect Ratio\600m\600x600m\results_50msec_100S_set2_primary",
                              output_folder=out_folder,
                              ar=True)

        print(f"Process complete after: {int(math.floor((time.time() - t) / 60)):02d}:{int(time.time() % 60):02d}.")

    def test_parsing(folder):
        """Try parsing every file in folder"""
        files = folder.glob("*.dat")
        for file in files:
            mun_file = MUNFile()
            mun_file.parse(file)

        print(f"Parsing complete.")

    samples_folder = Path(__file__).parents[2].joinpath('sample_files')
    convert_folders()
    # test_parsing(samples_folder.joinpath(r"Two-way induction\300x100\100S\MUN"))

    # file = r"A:\IRAP\All_3D_data_files\Aspect Ratio\150m\5x150m\results_50msec_1kS_set2.DAT"
    # mun_file = parser.parse(file)

    # folder = r"A:\IRAP\All_3D_data_files\Aspect Ratio\150m\150x150m"
    # primary_folder = r"A:\IRAP\All_3D_data_files\Aspect Ratio\150m\5x150m\results_50msec_100S_set2_primary"
    # mun_file = MUNFile()
    # mun_file.convert(folder, primary_folder)
