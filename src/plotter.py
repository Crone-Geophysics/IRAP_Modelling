import sys
import pandas as pd
import numpy as np
from src.file_types.fem_file import FEMFile
from src.file_types.tem_file import TEMFile
from src.file_types.platef_file import PlateFFile
from src.file_types.mun_file import MUNFile
from pathlib import Path
from PyQt5 import (QtGui, QtCore, uic)
from PyQt5.QtWidgets import (QMainWindow, QApplication, QMessageBox, QInputDialog, QErrorMessage, QFileDialog,
                             QLineEdit, QFormLayout, QWidget)

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


class IRAPFileTab(QWidget):

    def __init__(self):
        super().__init__()
        self.layout = QFormLayout()
        self.setLayout(self.layout)


    def read(self, filepath):
        ext = Path(filepath).suffix.lower()

        if ext == '.tem':
            parser = TEMFile()
            f = parser.parse(filepath)
        elif ext == '.fem':
            parser = FEMFile()
            f = parser.parse(filepath)
        elif ext == '.dat':
            first_line = open(filepath).readlines()[0]
            if 'Data type:' in first_line:
                parser = MUNFile()
                f = parser.parse(filepath)
            else:
                parser = PlateFFile()
                f = parser.parse(filepath)


class Plotter(QMainWindow, plotterUI):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setAcceptDrops(True)
        self.setWindowTitle("IRAP Plotter")
        # self.setWindowIcon(QtGui.QIcon(str(icons_path.joinpath('voltmeter.png'))))
        self.err_msg = QErrorMessage()
        self.msg = QMessageBox()

    def dragEnterEvent(self, e):
        e.accept()

    def dragMoveEvent(self, e):
        """
        Controls which files can be drag-and-dropped into the program.
        :param e: PyQT event
        """
        urls = [url.toLocalFile() for url in e.mimeData().urls()]
        if all([Path(file).suffix.lower() in ['.dat', '.tem', '.fem'] for file in urls]):
            print(f"Action accepted")
            e.acceptProposedAction()
            return
        else:
            print(f"Action rejected")
            e.ignore()

    def dropEvent(self, e):
        urls = [url.toLocalFile() for url in e.mimeData().urls()]
        self.open(urls[0])

    def open(self, filepath):
        path = Path(filepath)
        name = path.name
        ext = path.suffix.lower()

        if ext not in ['.tem', '.fem', '.dat']:
            self.msg.showMessage(self, 'Error', f"{ext[1:]} is not an implemented file extension.")
            print(f"{ext} is not supported.")

        print(f"Opening {name}.")
        tab = IRAPFileTab()
        self.tab_widget.addTab(tab, name)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    sample_files = Path(__file__).parents[1].joinpath('sample_files')

    mun_file = sample_files.joinpath(r'MUN files\LONG_V1x1_450_50_100_50msec_3D_solution_channels_tem_time_decay_z.dat')
    platef_file = sample_files.joinpath(r'PLATEF files\450_50.dat')
    fem_file = sample_files.joinpath(r'Maxwell files\Test 4 FEM files\Test 4 - h=5m.fem')
    tem_file = sample_files.joinpath(r'Maxwell files\V_1x1_450_50_100 50msec instant on-time first.tem')

    pl = Plotter()
    pl.show()
    #
    # pl.open(tem_file)
    # pl.open(fem_file)
    # pl.open(platef_file)
    # pl.open(mun_file)

    app.exec_()
