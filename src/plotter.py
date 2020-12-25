import sys
import os
import pandas as pd
import numpy as np
from src.file_types.fem_file import FEMFile
from src.file_types.tem_file import TEMFile
from src.file_types.platef_file import PlateFFile
from src.file_types.mun_file import MUNFile
from pathlib import Path
from PyQt5 import (QtGui, QtCore, uic)
from PyQt5.QtWidgets import (QMainWindow, QApplication, QMessageBox, QInputDialog, QErrorMessage, QFileDialog,
                             QLineEdit)

# Modify the paths for when the script is being run in a frozen state (i.e. as an EXE)
if getattr(sys, 'frozen', False):
    application_path = sys.executable
    plotter_ui_file = 'ui\\plotter.ui'
    icons_path = 'ui\\icons'
else:
    application_path = os.path.dirname(os.path.abspath(__file__))
    plotter_ui_file = os.path.join(application_path, 'ui\\plotter.ui')
    icons_path = os.path.join(application_path, "ui\\icons")
plotter_ui, _ = uic.loadUiType(plotter_ui_file)


class Plotter(QMainWindow, plotter_ui):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setAcceptDrops(True)
        self.setWindowTitle("IRAP Plotter")
        self.setWindowIcon(QtGui.QIcon(os.path.join(icons_path, 'icon.png')))
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
        if all([Path(file).suffix.lower() in ['.dat', '.tem'] for file in urls]):
            e.acceptProposedAction()
            return
        else:
            e.ignore()

    def dropEvent(self, e):
        urls = [url.toLocalFile() for url in e.mimeData().urls()]

    def open(self, file):
        ext = Path(file).suffix.lower()

        if ext == '.tem':
            parser = TEMFile()
            f = parser.parse(file)
        elif ext == '.fem':
            parser = FEMFile()
            f = parser.parse(file)
        elif ext == '.dat':
            first_line = open(file).readlines()[0]
            if 'Data type:' in first_line:
                parser = MUNFile()
                f = parser.parse(file)
            else:
                parser = PlateFFile()
                f = parser.parse(file)
        else:
            self.msg.showMessage(self, 'Error', f"{ext[1:]} is not an implemented file extension.")
            f = None

        return f


if __name__ == '__main__':
    app = QApplication(sys.argv)

    sample_files = Path(__file__).parents[1].joinpath('sample_files')

    mun_file = sample_files.joinpath(r'MUN files\LONG_V1x1_450_50_100_50msec_3D_solution_channels_tem_time_decay_z.dat')
    platef_file = sample_files.joinpath(r'PLATEF files\450_50.dat')
    fem_file = sample_files.joinpath(r'Maxwell files\Test 4 FEM files\Test 4 - h=5m.fem')
    tem_file = sample_files.joinpath(r'Maxwell files\V_1x1_450_50_100 50msec instant on-time first.tem')

    # pl = Plotter()
    # pl.show()
    #
    # pl.open(tem_file)
    # pl.open(fem_file)
    # pl.open(platef_file)
    # pl.open(mun_file)

    app.exec_()
