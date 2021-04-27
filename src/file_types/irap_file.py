from pathlib import Path

import numpy as np
import pandas as pd
import re
from PyQt5.QtWidgets import (QLabel)
from natsort import natsorted
from io import StringIO

from src.file_types.base_tdem_widget import BaseTDEM


class IRAPTab(BaseTDEM):

    def __init__(self, parent=None, axes=None):
        raise NotImplementedError()
        # super().__init__(parent=parent, axes=axes)
        # self.layout.insertRow(1, "File Type", QLabel("PlateF File"))
        #
        # self.color = "r"

    def read(self, filepath):
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        ext = filepath.suffix.lower()

        if ext == '.dat':
            parser = PlateFFile()
            try:
                file = parser.parse(filepath)
            except Exception as e:
                raise Exception(f"The following error occurred trying to parse the file: {e}.")
        else:
            raise ValueError(f"{ext} is not yet supported.")

        if file is None:
            raise ValueError(F"No data found in {filepath.name}.")

        # Add the file name as the default for the name in the legend
        # self.layout.addRow('Units', QLabel(file.units))
        self.layout.addRow('Current', QLabel(f"{float(file.current):,}"))

        self.layout.addRow('Rx Area', QLabel(str(file.rx_area)))

        if file.components:
            self.layout.addRow('Components', QLabel('\n'.join(natsorted(file.components))))

        self.layout.addRow(QLabel("Plot Channels"), self.ch_select_frame)

        channel_times = file.ch_times.astype(float) * 1000
        channel_times.index += 1
        self.layout.addRow('Channel Times', QLabel(channel_times.to_string()))

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
        self.legend_name.setText(f"{self.file.filepath.stem} (PLATE)")

    def plot(self):
        """
        Plot the data on a mpl axes
        """
        # Remove existing plotted lines
        self.clear()

        self.x_artists = []
        self.y_artists = []
        self.z_artists = []

        channels = [f'{num}' for num in range(1, len(self.file.ch_times) + 1)]
        plotting_channels = channels[self.min_ch.value() - 1: self.max_ch.value()]

        for component in self.file.components:
            comp_data = self.data[self.data.Component == component]

            if comp_data.empty:
                print(f"No {component} data in {self.file.filepath.name}.")
                continue

            size = 8  # For scatter point size

            ax = self.axes[component]

            for ind, ch in enumerate(plotting_channels):
                if ind == 0:
                    label = self.legend_name.text()
                else:
                    label = None

                x = comp_data.Station.astype(float) + self.shift_stations_sbox.value()
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


class IRAPFile:
    """
    IRAP file object
    """

    def __init__(self):
        self.filepath = None

        self.ch_times = pd.DataFrame()
        self.name = None
        self.x_dim = None
        self.y_dim = None
        self.conductance = None
        self.data = pd.DataFrame()
        self.components = []

    @staticmethod
    def convert(filepath):
        """
        Create a txt file for each model inside Peter's text file. Saves the files in the same directory.
        :param filepath: Path or str
        """
        filepath = Path(filepath)
        if not filepath.is_file():
            raise ValueError(f"{str(filepath)} is not a file.")

        print(f"Converting {filepath.name}")
        with open(filepath, 'r') as file:
            content = file.read()

        # waveform_match = re.split(r"Current waveform:", content)[-1].split("DONE")[0].split("\n")
        # waveform = [string.split() for string in waveform_match]
        # waveform = pd.DataFrame(waveform).dropna().iloc[:, 3].astype(float)

        ch_times_match = re.split(r"Gate times in order of output:", content)[-1].split("DONE")[0].split("\n")
        ch_times_text = pd.Series(np.concatenate([string.split() for string in ch_times_match]))
        ch_times_text = [re.search(r"\[(.*)\,(.*)\]", time).groups() for time in ch_times_text]
        ch_times = pd.DataFrame.from_records(ch_times_text, columns=["Start", "End"]).astype(float)
        # num_channels = len(waveform)  # Number of off-time channels

        # on_time = float(re.search("OnTime: (.*) PulseTime", content).group(1).strip())
        # pulse_time = float(re.search("PulseTime: (.*) ExponentialRiseConstant", content).group(1).strip())
        # survey_type = re.search("Sensor = (.*)", content).group(1).strip()

        # Data
        columns = np.insert(ch_times.index.astype(str), 0, ["Station", "Component"])
        model_matches = re.split(r"\$\$ MODEL", content)[1:]
        count = 1
        num_files = len(model_matches)
        for model_text in model_matches:
            model_text = model_text.strip()
            info = model_text.split("\n")[0]
            name = info.split(":")[0]
            conductance = float(re.search(r"Conductance = (.*);", info).group(1))
            x_dim, y_dim = re.search(r"xdim=(.*)ydim=(.*)", re.sub(r"[\s]", "", model_text.split("\n")[0])).groups()
            data_matches = model_text.split(r"###")[1:]
            model_data = pd.DataFrame()

            for data_text in data_matches:
                component = re.search(r"Outputting Rx component: \d =(\w)", data_text)
                if not component:
                    raise ValueError(f"No component found in {filepath.name}.")
                component = component.group(1).upper()
                readings = [arr.split() for arr in data_text.split("\n")[1:]]
                data_df = pd.DataFrame.from_records(readings).dropna().astype(float)
                data_df.insert(1, "Component", component)
                data_df.columns = columns
                model_data = model_data.append(data_df)

            head = f"Name:{name.upper()} X_Dim:{x_dim} Y_Dim:{y_dim} Conductance:{conductance}"
            ch_times_text = ch_times
            file_name = filepath.parent.joinpath(f"{x_dim}x{y_dim}{name.upper()}").with_suffix(".dat")
            file_text = F"{head}\n\n" \
                        F"### Channel Times ###\n" \
                        F"{ch_times_text}\n\n" \
                        F"### Data ###\n" \
                        F"{model_data.to_string(header=True, index=False)}"
            print(f"Saving {file_name} ({count}/{num_files}).")

            with open(file_name, 'w') as f:
                f.write(file_text)
            count += 1

    def parse(self, filepath):
        """
        Parse a custom text file (which was converted from Peter's format previously).
        :param filepath: Path or str
        """
        self.filepath = Path(filepath)

        if not self.filepath.is_file():
            raise ValueError(f"{self.filepath} is not a file.")

        print(f"Parsing {self.filepath.name}")
        with open(filepath, 'r') as file:
            content = file.read()

        head = content.split("\n")[0].strip().split()
        if head[0] == "":
            raise ValueError(F"{self.filepath.name} is not the correct file format.")

        self.name = re.sub(r"Name:", "", head[0])
        self.x_dim = re.sub(r"X_Dim:", "", head[1])
        self.y_dim = re.sub(r"Y_Dim:", "", head[2])
        self.conductance = re.sub(r"Conductance:", "", head[3])

        ch_times_text = content.split("### Channel Times ###")[1].split("### Data ###")[0].strip()
        ch_times_io = StringIO(ch_times_text)
        self.ch_times = pd.read_csv(ch_times_io, delim_whitespace=True)

        # Data
        data_text = content.split("### Data ###")[1].strip()
        data_io = StringIO(data_text)
        self.data = pd.read_csv(data_io, delim_whitespace=True)
        orgi_cols = [str(i) for i in range(len(self.ch_times))]
        new_cols = [str(i) for i in range(1, len(self.ch_times) + 1)]
        self.data.rename(columns=dict(zip(orgi_cols, new_cols)), inplace=True)
        self.components = self.data.Component.unique()

        return self

    def get_range(self):
        channels = self.ch_times.index
        data = self.data.loc[:, channels]
        mn = data.min().min()
        mx = data.max().max()
        print(f"Data range of {self.filepath.name} is {mn} to {mx}.")
        return mn, mx


if __name__ == '__main__':
    irap_file_parser = IRAPFile()

    sample_files = Path(__file__).parents[2].joinpath('sample_files')
    # file = sample_files.joinpath(r'Aspect ratio\Peter\2021-03-11_MUN_150m_ModelGroup.txt')
    # peter_file = peter_file_parser.convert(file)
    file = sample_files.joinpath(r'Aspect ratio\IRAP\50x150A.dat')
    irap_file = irap_file_parser.parse(file)
    # irap_file.get_range()
