from pathlib import Path

import numpy as np
import pandas as pd
import re
from PyQt5.QtWidgets import (QLabel)
from natsort import natsorted

from src.file_types.base_tdem_widget import BaseTDEM


class PeterTab(BaseTDEM):

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


class PeterFile:
    """
    Peter file object
    """

    def __init__(self):
        self.filepath = None

        self.ch_times = pd.DataFrame()
        self.on_time = None
        self.pulse_time = None
        self.survey_type = None
        self.units = 'nT/s'
        self.data = []
        self.components = []

    def convert(self, filepath):
        self.filepath = Path(filepath)

        if not self.filepath.is_file():
            raise ValueError(f"{self.filepath} is not a file.")

        print(f"Parsing {self.filepath.name}")
        with open(filepath, 'r') as file:
            content = file.read()
            # split_content = content.split('\n')

        waveform_match = re.split(r"Current waveform:", content)[-1].split("DONE")[0].split("\n")
        waveform = [string.split() for string in waveform_match]
        waveform = pd.DataFrame(waveform).dropna().iloc[:, 3].astype(float)

        ch_times_match = re.split(r"Gate times in order of output:", content)[-1].split("DONE")[0].split("\n")
        ch_times_text = pd.Series(np.concatenate([string.split() for string in ch_times_match]))
        ch_times_text = [re.search(r"\[(.*)\,(.*)\]", time).groups() for time in ch_times_text]
        ch_times = pd.DataFrame.from_records(ch_times_text, columns=["Start", "End"]).astype(float)
        num_channels = len(waveform)  # Number of off-time channels

        self.on_time = float(re.search("OnTime: (.*) PulseTime", content).group(1).strip())
        self.pulse_time = float(re.search("PulseTime: (.*) ExponentialRiseConstant", content).group(1).strip())
        self.survey_type = re.search("Sensor = (.*)", content).group(1).strip()

        # Data
        columns = np.insert(ch_times.index.astype(str), 0, ["Station", "Component"])
        model_matches = re.split(r"\$\$ MODEL", content)[1:]
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
                    raise ValueError(f"No component found in {self.filepath.name}.")
                component = component.group(1).upper()
                readings = [arr.split() for arr in data_text.split("\n")[1:]]
                data_df = pd.DataFrame.from_records(readings).dropna().astype(float)
                data_df.insert(1, "Component", component)
                data_df.columns = columns
                model_data = model_data.append(data_df)

            # data = {"Name": name.upper(), "X_Dim": x_dim, "Y_Dim": y_dim, "Conductance": conductance, "Data": model_data}
            # self.data.append(data)
            head = f"Name: {name.upper()} X_Dim: {x_dim} Y_Dim: {y_dim} Conductance: {conductance}\n"
            ch_times = ch_times.to_numpy()
            file_name = self.filepath.parent.joinpath(f"{x_dim}x{y_dim}{name.upper()}").with_suffix(".pete")
            print(f"Saving {file_name}.")

            with open(file_name, 'a') as f:
                f.write(head)
                f.write(model_data.to_string(header=True, index=False))
            # model_data.to_csv(file_name, index=False)

        # self.components = np.unique(np.concatenate(
        #     [data.Component.unique() for data in [lst["Data"] for lst in self.data]]
        # ))
        # return self

    def get_range(self):
        channels = self.ch_times.index
        all_data = [lst["Data"] for lst in self.data]
        data = self.data.loc[:, channels]
        mn = data.min().min()
        mx = data.max().max()
        return mn, mx


if __name__ == '__main__':
    peter_file_parser = PeterFile()

    sample_files = Path(__file__).parents[2].joinpath('sample_files')
    file = sample_files.joinpath(r'Aspect ratio\Peter\2021-03-11_MUN_150m_ModelGroup.txt')
    peter_file = peter_file_parser.convert(file)
    # peter_file.get_range()
