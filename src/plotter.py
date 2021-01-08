import sys
import os
import pickle
import io
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from natsort import natsorted

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.pyplot import cm

from src.file_types.fem_file import FEMFile, FEMTab
from src.file_types.tem_file import TEMFile, TEMTab
from src.file_types.platef_file import PlateFFile, PlateFTab
from src.file_types.mun_file import MUNFile, MUNTab
from PyQt5 import (QtCore, uic)
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

matplotlib.use('Qt5Agg')
# color = iter(cm.gist_rainbow(np.linspace(0, 1, 20)))
color = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
# color = iter(cm.tab10())


class Plotter(QMainWindow, plotterUI):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setAcceptDrops(True)
        self.setWindowTitle("IRAP Plotter")
        # self.setWindowIcon(QtGui.QIcon(str(icons_path.joinpath('voltmeter.png'))))
        self.err_msg = QErrorMessage()
        self.msg = QMessageBox()
        self.opened_files = []

        # Figure
        self.fem_figure = Figure()
        self.fem_figure.set_size_inches((11, 8.5))
        self.ax = self.fem_figure.add_subplot(111)
        self.ax.set_xlabel("Station")
        self.fem_canvas = FigureCanvas(self.fem_figure)

        toolbar = NavigationToolbar(self.fem_canvas, self)

        self.canvas_frame.layout().addWidget(self.fem_canvas)
        self.canvas_frame.layout().addWidget(toolbar)

        # Signals
        self.actionOpen.triggered.connect(self.open_file_dialog)
        self.actionPrint_to_PDF.triggered.connect(self.print_pdf)

        self.tab_widget.tabCloseRequested.connect(self.update_tab_plot)

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Space:
            print(f"Replotting")
            self.ax.relim()
            self.ax.autoscale()

            self.fem_canvas.draw()
            self.fem_canvas.flush_events()

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

    def print_pdf(self):
        """Resize the figure to 11 x 8.5" and save to a PDF file"""
        filepath, ext = QFileDialog.getSaveFileName(self, 'Save PDF', '', "PDF Files (*.PDF);;All Files (*.*)")

        if filepath:
            # Create a copy of the figure
            buf = io.BytesIO()
            pickle.dump(self.fem_figure, buf)
            buf.seek(0)
            save_figure = pickle.load(buf)

            # Resize and save the figure
            save_figure.set_size_inches((11, 8.5))
            save_figure.savefig(filepath,
                                orientation='landscape')

            self.statusBar().showMessage(f"PDF saved to {filepath}.", 1500)
            os.startfile(filepath)

    def open_file_dialog(self):
        """Open files through the file dialog"""
        filepaths, ext = QFileDialog.getOpenFileNames(None, "Open FEM File", "",
                                                      "Maxwell FEM Files (*.FEM);;All Files (*.*)")

        if filepaths:
            for file in filepaths:
                self.open(file)

    def open(self, filepath):
        """
        Read and plot a FEM file.
        :param filepath: str or Path object
        """

        def update_legend_label(tab):
            """Update the legend name for the lines in the tab"""

            for line in tab.lines:
                line._label = "updated"

            self.fem_canvas.draw()
            self.fem_canvas.flush_events()
            self.update_legend()

        filepath = Path(filepath)
        ext = filepath.suffix.lower()

        if ext not in ['.tem', '.fem', '.dat']:
            self.msg.showMessage(self, 'Error', f"{ext[1:]} is not an implemented file extension.")
            print(f"{ext} is not supported.")
            return

        elif filepath in self.opened_files:
            print(f"{filepath.name} is already opened.")
            return

        print(f"Opening {filepath.name}.")

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
            tab.read(filepath)
        except Exception as e:
            self.err_msg.showMessage(str(e))
            return

        # Connect signals
        tab.show_sig.connect(lambda: self.update_tab_plot(tab))
        # tab.legend_name.editingFinished.connect(lambda: update_legend_label(tab))

        self.tab_widget.addTab(tab, filepath.name)

        # Plot the data from the file
        c = next(color)  # Cycles through colors
        tab.plot(self.ax, c)

        # Add the Y axis label
        if not self.ax.get_ylabel() or self.tab_widget.count() == 1:
            self.ax.set_ylabel(tab.file.units)
        else:
            if self.ax.get_ylabel() != tab.file.units:
                self.msg.warning(self, "Warning", f"The units for {tab.file.filepath.name} are"
                                                  f" different then the prior units.")

        # Update the plot
        self.fem_canvas.draw()
        self.fem_canvas.flush_events()
        self.update_legend()

        self.opened_files.append(filepath)

    def update_legend(self):
        """Update the legend to be in alphabetical order"""

        # Only sort if there are tabs, otherwise it crashes.
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            # sort both labels and handles by labels
            labels, handles = zip(*natsorted(zip(labels, handles), key=lambda t: t[0]))
            self.ax.legend(handles, labels)
        else:
            self.ax.legend()

        self.fem_canvas.draw()
        self.fem_canvas.flush_events()

    def update_tab_plot(self, tab):
        print(f"Updating plot")
        ind = None

        # Find the tab when an index is passed (when a tab is closed)
        if isinstance(tab, int):
            ind = tab
            tab = self.tab_widget.widget(ind)
            self.opened_files.pop(ind)

        artists = tab.lines

        lines = list(filter(lambda x: isinstance(x, matplotlib.lines.Line2D), tab.lines))
        points = list(filter(lambda x: isinstance(x, matplotlib.collections.PathCollection), tab.lines))  # Scatters

        if lines:
            # Add or remove the lines
            if all([artist in self.ax.lines for artist in artists]):
                for artist in artists:
                    self.ax.lines.remove(artist)
            else:
                if ind is None:  # Only add lines if the tab isn't being removed
                    for artist in artists:
                        self.ax.lines.append(artist)

        if points:
            # Add or remove the scatter points
            if all([artist in self.ax.collections for artist in artists]):
                for artist in artists:
                    self.ax.collections.remove(artist)
            else:
                if ind is None:  # Only add lines if the tab isn't being removed
                    for artist in artists:
                        self.ax.collections.append(artist)

        if ind is not None:
            self.tab_widget.removeTab(ind)

        # Update the plot
        self.update_legend()
        # self.fem_canvas.draw()
        # self.fem_canvas.flush_events()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    sample_files = Path(__file__).parents[1].joinpath('sample_files')

    mun_file = sample_files.joinpath(r'MUN files\LONG_V1x1_450_50_100_50msec_3D_solution_channels_tem_time_decay_z.dat')
    platef_file = sample_files.joinpath(r'PLATEF files\450_50.dat')
    fem_file = sample_files.joinpath(r'Maxwell files\FEM\Horizontal Plate 100S Normalized.fem')
    # fem_file = sample_files.joinpath(r'Maxwell files\FEM\Test 4 FEM files\Test 4 - h=5m.fem')
    # fem_file = sample_files.joinpath(r'Maxwell files\FEM\Turam 2x4 608S_0.96691A_PFCALC at 1A.fem')
    tem_file = sample_files.joinpath(r'Maxwell files\V_1x1_450_50_100 50msec instant on-time first.tem')

    pl = Plotter()
    pl.show()

    pl.open(fem_file)

    # pl.open(tem_file)
    # pl.open(fem_file)
    # pl.open(platef_file)
    # pl.open(mun_file)

    app.exec_()
