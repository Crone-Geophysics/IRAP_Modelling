import matplotlib
import pandas as pd
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (QLabel, QFormLayout, QWidget, QCheckBox, QFrame, QHBoxLayout, QSpinBox, QDoubleSpinBox,
                             QSizePolicy, QLineEdit)


class BaseTDEM(QWidget):
    plot_changed_sig = QtCore.pyqtSignal()

    def __init__(self, parent=None, axes=None):
        """
        Base widget to be inherited by other TDEM files.
        :param parent: Qt widget parent
        :param axes: dict of axes for each component to be plotted on.
        """
        super().__init__()
        self.layout = QFormLayout()
        self.setLayout(self.layout)

        self.plot_cbox = QCheckBox("Plot")
        self.plot_cbox.setFocusPolicy(QtCore.Qt.NoFocus)
        self.plot_cbox.setChecked(True)

        self.layout.addRow(self.plot_cbox)

        self.legend_name = QLineEdit()
        self.legend_name.setFixedWidth(120)
        self.layout.addRow("Legend Name", self.legend_name)

        # Data editing
        self.scale_data_sbox = QDoubleSpinBox()
        self.scale_data_sbox.setValue(1.)
        self.scale_data_sbox.setSingleStep(0.1)
        self.scale_data_sbox.setDecimals(7)
        self.scale_data_sbox.setMaximum(1e9)
        self.scale_data_sbox.setMinimum(-1e9)
        self.scale_data_sbox.setGroupSeparatorShown(True)
        self.scale_data_sbox.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.scale_data_sbox.setFixedWidth(120)
        self.layout.addRow("Scale Data", self.scale_data_sbox)

        self.shift_stations_sbox = QDoubleSpinBox()
        self.shift_stations_sbox.setMaximum(100000)
        self.shift_stations_sbox.setMinimum(-100000)
        self.shift_stations_sbox.setGroupSeparatorShown(True)
        self.shift_stations_sbox.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.shift_stations_sbox.setFixedWidth(120)
        self.layout.addRow("Shift Stations", self.shift_stations_sbox)

        self.alpha_sbox = QSpinBox()
        self.alpha_sbox.setSingleStep(10)
        self.alpha_sbox.setRange(0, 100)
        self.alpha_sbox.setValue(100)
        self.alpha_sbox.setSuffix('%')
        self.alpha_sbox.setFixedWidth(120)
        self.layout.addRow("Plot Alpha", self.alpha_sbox)

        # Channel selection frame
        self.ch_select_frame = QFrame()
        self.ch_select_frame.setLayout(QHBoxLayout())
        self.ch_select_frame.layout().setContentsMargins(0, 0, 0, 0)
        self.ch_select_frame.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.min_ch = QSpinBox()
        self.max_ch = QSpinBox()
        self.min_ch.setMinimum(1)
        self.max_ch.setMinimum(1)
        self.min_ch.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.max_ch.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        # self.ch_select_frame.layout().addWidget(QLabel("Plot Channels"))
        self.ch_select_frame.layout().addWidget(self.min_ch)
        self.ch_select_frame.layout().addWidget(QLabel("to"))
        self.ch_select_frame.layout().addWidget(self.max_ch)

        self.file = None
        self.parent = parent
        self.color = None
        self.alpha = None  # Remember the last alpha
        self.axes = axes

        self.x_artists = []
        self.y_artists = []
        self.z_artists = []
        self.data = pd.DataFrame()

        # Signals
        self.plot_cbox.toggled.connect(self.toggle)
        self.legend_name.editingFinished.connect(self.plot)
        self.scale_data_sbox.valueChanged.connect(self.plot)
        self.scale_data_sbox.valueChanged.connect(lambda: self.plot_changed_sig.emit())
        self.shift_stations_sbox.valueChanged.connect(self.plot)
        self.shift_stations_sbox.valueChanged.connect(lambda: self.plot_changed_sig.emit())
        self.alpha_sbox.valueChanged.connect(self.plot)
        self.alpha_sbox.valueChanged.connect(lambda: self.plot_changed_sig.emit())
        self.min_ch.valueChanged.connect(lambda: self.update_channels("min"))
        self.max_ch.valueChanged.connect(lambda: self.update_channels("max"))

    def plot(self):
        """
        Plot the data on a mpl axes.
        """
        pass

    def clear(self):
        # Remove existing plotted lines
        for ls, ax in zip([self.x_artists, self.y_artists, self.z_artists], self.axes.values()):
            if ax.lines:
                if all([artist in ax.lines for artist in ls]):
                    for artist in ls:
                        ax.lines.remove(artist)
            if ax.collections:
                if all([artist in ax.collections for artist in ls]):
                    for artist in ls:
                        ax.collections.remove(artist)

        self.x_artists = []
        self.y_artists = []
        self.z_artists = []

    def toggle(self):
        """Toggle the visibility of plotted lines/points"""
        for ax in self.axes.values():
            if ax == self.axes['X']:
                artists = self.x_artists
            elif ax == self.axes['Y']:
                artists = self.y_artists
            else:
                artists = self.z_artists

            lines = list(filter(lambda x: isinstance(x, matplotlib.lines.Line2D), artists))
            points = list(filter(lambda x: isinstance(x, matplotlib.collections.PathCollection), artists))  # Scatters

            if lines:
                if self.plot_cbox.isChecked():
                    if all([a in ax.lines for a in lines]):  # If the lines are already plotted, pass.
                        pass
                    else:
                        for artist in artists:
                            ax.lines.append(artist)
                else:
                    for artist in artists:
                        ax.lines.remove(artist)

            if points:
                # Add or remove the scatter points
                if self.plot_cbox.isChecked():
                    if all([a in ax.collections for a in points]):  # If the points are already plotted, pass.
                        pass
                    else:
                        for artist in artists:
                            ax.collections.append(artist)
                else:
                    for artist in artists:
                        ax.collections.remove(artist)

        self.plot_changed_sig.emit()

    def update_channels(self, which):
        """
        Change the plotted channel range
        :param which: str, which channel extreme was changed, min or max
        """
        self.min_ch.blockSignals(True)
        self.max_ch.blockSignals(True)

        min_ch = self.min_ch.value()
        max_ch = self.max_ch.value()

        if which == 'min':
            if min_ch > max_ch:
                self.max_ch.setValue(min_ch)

        else:
            if max_ch < min_ch:
                self.min_ch.setValue(max_ch)

        self.plot()

        self.min_ch.blockSignals(False)
        self.max_ch.blockSignals(False)

        for ax in self.axes.values():
            # Re-scale the plots
            ax.relim()
            ax.autoscale()

