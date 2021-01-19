import sys
from PyQt5.QtWidgets import (QApplication)
from src.plotter import TEMPlotter

if __name__ == '__main__':
    app = QApplication(sys.argv)

    plotter = TEMPlotter()
    plotter.show()

    app.exec_()
