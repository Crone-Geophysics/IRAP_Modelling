import sys
import os
import pandas as pd
import numpy as np
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


def parse_maxwell_tem(file):
    print(f"Parsing Maxwell TEM file.")

    with open(file, 'r') as f:
        content = f.read()

    return None


def parse_maxwell_fem(file):
    print(f"Parsing Maxwell FEM file.")

    with open(file, 'r') as f:
        content = f.read()

    return None


def parse_mun(file):
    print(f"Parsing MUN file.")
    return None


def parse_platef(file):
    print(f"Parsing PLATEF file.")
    return None


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
            f = parse_maxwell_tem(file)
        elif ext == '.fem':
            f = parse_maxwell_fem(file)
        elif ext == '.dat':
            first_line = open(file).readlines()[0]
            if 'Data type:' in first_line:
                f = parse_mun(file)
            else:
                f = parse_platef(file)
        else:
            self.msg.showMessage(self, 'Error', f"{ext[1:]} is not an implemented file extension.")
            return


if __name__ == '__main__':
    app = QApplication(sys.argv)

    maxwell_tem_file = r'C:\Users\Eric\PycharmProjects\IRAP_Modelling\sample_files\Maxwell files\V_1x1_450_50_100 50msec instant on-time first.tem'
    maxwell_fem_file = r'C:\Users\Eric\PycharmProjects\IRAP_Modelling\sample_files\Maxwell files\Test #2.fem'
    mun_file = r'C:\Users\Eric\PycharmProjects\IRAP_Modelling\sample_files\MUN files\LONG_V1x1_450_50_100_50msec_3D_solution_channels_tem_time_decay_z.dat'
    platef_file = r'C:\Users\Eric\PycharmProjects\IRAP_Modelling\sample_files\PLATEF files\450_50.dat'

    pl = Plotter()
    pl.show()

    pl.open(maxwell_tem_file)
    pl.open(maxwell_fem_file)
    pl.open(mun_file)
    pl.open(platef_file)

    app.exec_()
