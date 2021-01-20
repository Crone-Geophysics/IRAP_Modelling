import sys
from PyQt5.QtWidgets import (QApplication)
from src.plotter import FEMPlotter

if __name__ == '__main__':
    app = QApplication(sys.argv)

    plotter = FEMPlotter()
    plotter.show()

    app.exec_()
