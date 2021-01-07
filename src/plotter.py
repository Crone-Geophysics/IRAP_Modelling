import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from src.file_types.fem_file import FEMFile, FEMTab
from src.file_types.tem_file import TEMFile, TEMTab
from src.file_types.platef_file import PlateFFile, PlateFTab
from src.file_types.mun_file import MUNFile, MUNTab
from PyQt5 import (QtGui, QtCore, uic)
from PyQt5.QtWidgets import (QMainWindow, QApplication, QMessageBox, QInputDialog, QErrorMessage, QFileDialog,
                             QLineEdit, QFormLayout, QWidget, QCheckBox)

# Modify the paths for when the script is being run in a frozen state (i.e. as an EXE)
if getattr(sys, 'frozen', False):
    application_path = Path(sys.executable).parent
    PlotterUIFile = Path('ui\\plotter.ui')
    icons_path = Path('ui\\icons')
else:
    application_path = Path(__file__).absolute().parent
    PlotterUIFile = application_path.joinpath('ui\\plotter.ui')
    icons_path = application_path.joinpath('ui\\icons')

# Load Qt ui file into a class
plotterUI, _ = uic.loadUiType(PlotterUIFile)


class Plotter(QMainWindow, plotterUI):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setAcceptDrops(True)
        self.setWindowTitle("IRAP Plotter")
        # self.setWindowIcon(QtGui.QIcon(str(icons_path.joinpath('voltmeter.png'))))
        self.err_msg = QErrorMessage()
        self.msg = QMessageBox()

        # Figure
        self.fem_figure = Figure(figsize=(8.5, 11))
        self.ax = self.fem_figure.add_subplot(111)
        self.ax.set_xlabel("Station")
        self.fem_canvas = FigureCanvas(self.fem_figure)

        toolbar = NavigationToolbar(self.fem_canvas, self)

        self.canvas_layout.addWidget(self.fem_canvas)
        self.canvas_layout.addWidget(toolbar)

        # Signals
        self.tab_widget.tabCloseRequested.connect(self.update_tab_plot)

    def dragEnterEvent(self, e):
        e.accept()

    def dragMoveEvent(self, e):
        """
        Controls which files can be drag-and-dropped into the program.
        :param e: PyQT event
        """
        urls = [url.toLocalFile() for url in e.mimeData().urls()]
        if all([Path(file).suffix.lower() in ['.dat', '.tem', '.fem'] for file in urls]):
            e.acceptProposedAction()
            return
        else:
            e.ignore()

    def dropEvent(self, e):
        urls = [url.toLocalFile() for url in e.mimeData().urls()]
        for url in urls:
            self.open(url)

    def open(self, filepath):
        path = Path(filepath)
        name = path.name
        ext = path.suffix.lower()

        if ext not in ['.tem', '.fem', '.dat']:
            self.msg.showMessage(self, 'Error', f"{ext[1:]} is not an implemented file extension.")
            print(f"{ext} is not supported.")
            return

        print(f"Opening {name}.")

        # Create a new tab and add it to the widget
        if ext == '.tem':
            tab = TEMTab()
        elif ext == '.fem':
            tab = FEMTab()
        elif ext == '.dat':
            first_line = open(filepath).readlines()[0]
            if 'Data type:' in first_line:
                tab = MUNTab()
            else:
                tab = PlateFTab()
        else:
            tab = None

        try:
            tab.read(path)
        except Exception as e:
            self.err_msg.showMessage(str(e))
            return

        tab.show_sig.connect(lambda: self.update_tab_plot(tab))
        self.tab_widget.addTab(tab, name)

        # Plot the data from the file
        tab.plot(self.ax)

        # Add the Y axis label
        if not self.ax.get_ylabel():
            self.ax.set_ylabel(tab.f.units)
        else:
            if self.ax.get_ylabel is not tab.f.units:
                self.msg.warning(self, "Warning", f"The units for {tab.f.filepath.name} are"
                                                  f" different then the prior units.")

        # Update the plot
        self.fem_canvas.draw()
        self.fem_canvas.flush_events()
        self.update_legend()

    def update_legend(self):
        """Update the legend to be in alphabetical order"""
        handles, labels = self.ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        self.ax.legend(handles, labels)

    def update_tab_plot(self, tab):
        print(f"Updating plot")
        ind = None

        # Find the tab when an index is passed (when a tab is closed)
        if isinstance(tab, int):
            ind = tab
            tab = self.tab_widget.widget(ind)

        lines = tab.lines

        # Add or remove the lines
        if all([line in self.ax.lines for line in lines]):
            for line in lines:
                self.ax.lines.remove(line)
        else:
            if ind is None:  # Only add lines if the tab isn't being removed
                for line in lines:
                    self.ax.lines.append(line)

        # Update the plot
        self.update_legend()
        self.fem_canvas.draw()
        self.fem_canvas.flush_events()

        if ind is not None:
            self.tab_widget.removeTab(ind)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    sample_files = Path(__file__).parents[1].joinpath('sample_files')

    mun_file = sample_files.joinpath(r'MUN files\LONG_V1x1_450_50_100_50msec_3D_solution_channels_tem_time_decay_z.dat')
    platef_file = sample_files.joinpath(r'PLATEF files\450_50.dat')
    fem_file = sample_files.joinpath(r'Maxwell files\Test 4 FEM files\Test 4 - h=5m.fem')
    # fem_file = sample_files.joinpath(r'Maxwell files\Turam 2x4 608S_0.96691A_PFCALC at 1A.fem')
    tem_file = sample_files.joinpath(r'Maxwell files\V_1x1_450_50_100 50msec instant on-time first.tem')

    pl = Plotter()
    pl.show()

    pl.open(fem_file)
    #
    # pl.open(tem_file)
    # pl.open(fem_file)
    # pl.open(platef_file)
    # pl.open(mun_file)

    app.exec_()
