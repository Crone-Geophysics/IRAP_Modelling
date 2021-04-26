import io
import math
import os
import pickle
import re
import sys
import time
import copy
from itertools import zip_longest
from pathlib import Path
from cycler import cycler

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5 import (QtCore, QtGui, uic)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMessageBox, QFrame, QErrorMessage, QFileDialog,
                             QTableWidgetItem, QScrollArea, QSpinBox, QHBoxLayout, QLabel, QInputDialog, QLineEdit,
                             QProgressDialog, QWidget, QHeaderView, QPushButton, QColorDialog)
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.pyplot import cm
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from natsort import natsorted, os_sorted
from scipy.signal import savgol_filter

from src.file_types.fem_file import FEMTab
from src.file_types.irap_file import IRAPFile
from src.file_types.mun_file import MUNFile, MUNTab
from src.file_types.platef_file import PlateFFile, PlateFTab
from src.file_types.tem_file import TEMFile, TEMTab

log_file_path = r"log.txt"
logging_file = open(log_file_path, "w+")

# Modify the paths for when the script is being run in a frozen state (i.e. as an EXE)
if getattr(sys, 'frozen', False):
    application_path = Path(sys.executable).parent
    FEMPlotterUIFile = Path('ui\\fem_plotter.ui')
    TEMPlotterUIFile = Path('ui\\tem_plotter.ui')
    TestRunnerUIFile = Path('ui\\test_runner.ui')
    icons_path = Path('ui\\icons')
else:
    application_path = Path(__file__).absolute().parent
    FEMPlotterUIFile = application_path.joinpath('ui\\fem_plotter.ui')
    TEMPlotterUIFile = application_path.joinpath('ui\\tem_plotter.ui')
    TestRunnerUIFile = application_path.joinpath('ui\\test_runner.ui')
    icons_path = application_path.joinpath('ui\\icons')

# Load Qt ui file into a class
fem_plotterUI, _ = uic.loadUiType(FEMPlotterUIFile)
tem_plotterUI, _ = uic.loadUiType(TEMPlotterUIFile)
test_runnerUI, _ = uic.loadUiType(TestRunnerUIFile)

matplotlib.use('Qt5Agg')
rainbow_colors = iter(cm.rainbow(np.linspace(0, 1, 20)))
quant_colors = np.nditer(np.array(plt.rcParams['axes.prop_cycle'].by_key()['color']))

# iter_colors = np.nditer(quant_colors)
# quant_colors = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
# color = iter(cm.tab10())

options = {"Maxwell": "*.TEM", "MUN": "*.DAT", "IRAP": "*.DAT", "PLATE": "*.DAT"}
colors = {"Maxwell": '#0000FF', "MUN": '#63DF48', "IRAP": "#000000", "PLATE": '#FF0000'}
styles = {"Maxwell": '-', "MUN": ":", "IRAP": "--", "PLATE": '-.'}


class ColorButton(QPushButton):
    """
    Custom Qt Widget to show a chosen color.

    Left-clicking the button shows the color-chooser, while
    right-clicking resets the color to None (no-color).
    """

    colorChanged = QtCore.pyqtSignal(object)

    def __init__(self, *args, color=None, **kwargs):
        super(ColorButton, self).__init__(*args, **kwargs)

        self.setObjectName("btn")  # Add name so when the button is colored, the QColorDialog won't change with it.
        self._color = None
        self._default = color
        self.pressed.connect(self.onColorPicker)

        # Set the initial/default state.
        self.setColor(self._default)

    def setColor(self, color):
        if color != self._color:
            self._color = color
            self.colorChanged.emit(color)

        if self._color:
            self.setStyleSheet("QPushButton#btn"
                               "{"
                               f"background-color: {self._color};"
                               "}")
        else:
            self.setStyleSheet("")

    def color(self):
        return self._color

    def onColorPicker(self):
        dlg = QColorDialog(self)
        if self._color:
            dlg.setCurrentColor(QtGui.QColor(self._color))

        if dlg.exec_():
            self.setColor(dlg.currentColor().name())

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.RightButton:
            self.setColor(self._default)

        return super(ColorButton, self).mousePressEvent(e)


class FEMPlotter(QMainWindow, fem_plotterUI):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setAcceptDrops(True)
        self.setWindowTitle("FEM Plotter v0.0")
        self.resize(800, 600)
        self.setWindowIcon(QtGui.QIcon(str(icons_path.joinpath('fem_plotter.png'))))
        self.err_msg = QErrorMessage()
        self.msg = QMessageBox()
        self.opened_files = []

        # HCP Figure
        self.hcp_figure = Figure()
        self.hcp_ax = self.hcp_figure.add_subplot(111)
        self.hcp_ax.set_xlabel("Station")
        self.hcp_ax.set_title("HCP Component")
        self.hcp_canvas = FigureCanvas(self.hcp_figure)

        toolbar = NavigationToolbar(self.hcp_canvas, self)

        self.hcp_canvas_frame.layout().addWidget(self.hcp_canvas)
        self.hcp_canvas_frame.layout().addWidget(toolbar)

        # VCA Figure
        self.vca_figure = Figure()
        self.vca_ax = self.vca_figure.add_subplot(111)
        self.vca_ax.set_xlabel("Station")
        self.vca_ax.set_title("VCA Component")
        self.vca_canvas = FigureCanvas(self.vca_figure)

        toolbar = NavigationToolbar(self.vca_canvas, self)

        self.vca_canvas_frame.layout().addWidget(self.vca_canvas)
        self.vca_canvas_frame.layout().addWidget(toolbar)

        self.axes = [self.hcp_ax, self.vca_ax]
        self.canvases = [self.hcp_canvas, self.vca_canvas]

        # Status bar
        self.num_files_label = QLabel()

        self.alpha_frame = QFrame()
        self.alpha_frame.setLayout(QHBoxLayout())
        self.alpha_frame.layout().addWidget(QLabel("Plot Alpha: "))
        self.alpha_frame.layout().setContentsMargins(0, 0, 0, 0)
        self.alpha_frame.setMaximumWidth(150)

        self.alpha_sbox = QSpinBox()
        self.alpha_sbox.setSingleStep(10)
        self.alpha_sbox.setRange(0, 100)
        self.alpha_sbox.setValue(100)

        self.alpha_frame.layout().addWidget(self.alpha_sbox)

        self.statusBar().addPermanentWidget(self.num_files_label, 1)
        self.statusBar().addPermanentWidget(self.alpha_frame)

        # Signals
        self.actionOpen.triggered.connect(self.open_file_dialog)
        self.actionPrint_to_PDF.triggered.connect(self.print_pdf)
        self.actionPlot_Legend.triggered.connect(self.update_legend)

        self.file_tab_widget.tabCloseRequested.connect(self.remove_tab)
        self.alpha_sbox.valueChanged.connect(self.update_alpha)

        self.update_num_files()

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Space:
            for canvas, ax in zip(self.canvases, self.axes):
                ax.relim()
                ax.autoscale()

                canvas.draw()
                canvas.flush_events()

    def dragEnterEvent(self, e):
        e.accept()

    def dragMoveEvent(self, e):
        """
        Controls which files can be drag-and-dropped into the program.
        :param e: PyQT event
        """
        urls = [url.toLocalFile() for url in e.mimeData().urls()]
        if all([Path(file).suffix.lower() in ['.fem'] for file in urls]):
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

        if not any([self.hcp_ax.lines, self.vca_ax.lines]):
            self.statusBar().showMessage(f"The plots are empty.", 1500)
            print(f"The plots are empty.")
            return

        filepath, ext = QFileDialog.getSaveFileName(self, 'Save PDF', '', "PDF Files (*.PDF);;All Files (*.*)")

        if filepath:
            with PdfPages(filepath) as pdf:
                # Print every figure as a PDF page
                for figure in [self.hcp_figure, self.vca_figure]:

                    # Only print the figure if there are plotted lines
                    if figure.axes[0].lines:
                        # Create a copy of the figure
                        buf = io.BytesIO()
                        pickle.dump(figure, buf)
                        buf.seek(0)
                        save_figure = pickle.load(buf)

                        # Resize and save the figure
                        save_figure.set_size_inches((11, 8.5))
                        pdf.savefig(save_figure, orientation='landscape')

            self.statusBar().showMessage(f"PDF saved to {filepath}.", 1500)
            # os.startfile(filepath)

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
        filepath = Path(filepath)
        ext = filepath.suffix.lower()

        if ext not in ['.fem']:
            self.msg.showMessage(self, 'Error', f"{ext[1:]} is not an implemented file extension.")
            print(f"{ext} is not supported.")
            return

        elif filepath in self.opened_files:
            print(f"{filepath.name} is already opened.")
            return

        print(f"Opening {filepath.name}.")

        try:
            color = str(next(quant_colors))  # Cycles through colors
        except StopIteration:
            print(f"Resetting color iterator.")
            quant_colors.reset()
            color = str(next(quant_colors))

        # Create a dict for which axes components get plotted on
        axes = {'HCP': self.hcp_ax, 'VCA': self.vca_ax}

        # Create a new tab and add it to the widget
        if ext == '.fem':
            tab = FEMTab(parent=self, color=color, axes=axes)
        else:
            tab = None

        try:
            tab.read(filepath)
        except Exception as e:
            self.err_msg.showMessage(str(e))
            return

        # Connect signals
        tab.plot_changed_sig.connect(self.update_legend)
        tab.plot_changed_sig.connect(self.update_ax_scales)

        # Create a new tab and add a scroll area to it, where the file tab is added to
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(250)
        scroll.setWidget(tab)
        self.file_tab_widget.addTab(scroll, filepath.name)

        self.plot_tab(tab)

        self.opened_files.append(filepath)
        self.update_num_files()

    def plot_tab(self, tab):
        # Find the tab when an index is passed (when re-plotting)
        if isinstance(tab, int):
            ind = tab
            tab = self.file_tab_widget.widget(ind).widget()

        alpha = self.alpha_sbox.value() / 100

        tab.plot(alpha)

        for canvas, ax in zip(self.canvases, self.axes):
            # Add the Y axis label
            if not ax.get_ylabel() or self.file_tab_widget.count() == 1:
                ax.set_ylabel(tab.file.units)
            else:
                if ax.get_ylabel() != tab.file.units:
                    print(f"Warning: The units for {tab.file.filepath.name} are different then the prior units.")
                    # self.msg.warning(self, "Warning", f"The units for {tab.file.filepath.name} are"
                    #                                   f" different then the prior units.")

            # Update the plot
            canvas.draw()
            canvas.flush_events()

        self.update_legend()

    def remove_tab(self, ind):
        """Remove a tab"""
        # Find the tab when an index is passed (when a tab is closed)
        tab = self.file_tab_widget.widget(ind).widget()
        tab.clear()
        self.opened_files.pop(ind)
        self.file_tab_widget.removeTab(ind)

        self.update_legend()
        self.update_num_files()

    def update_legend(self):
        """Update the legend to be in alphabetical order"""
        for canvas, ax in zip(self.canvases, self.axes):
            if self.actionPlot_Legend.isChecked():
                # Only sort if there are tabs, otherwise it crashes.
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    # sort both labels and handles by labels
                    labels, handles = zip(*natsorted(zip(labels, handles), key=lambda t: t[0]))
                    ax.legend(handles, labels).set_draggable(True)
                else:
                    ax.legend().set_draggable(True)
            else:
                legend = ax.get_legend()
                if legend:
                    legend.remove()

            canvas.draw()
            canvas.flush_events()

    def update_ax_scales(self):
        """Auto re-scale every plot"""
        for ax in self.axes:
            ax.relim()
            ax.autoscale()

        for canvas in self.canvases:
            canvas.draw()
            canvas.flush_events()

    def update_alpha(self, alpha):
        print(f"New alpha: {alpha / 100}")
        for canvas, ax in zip(self.canvases, self.axes):

            for artist in ax.lines:
                artist.set_alpha(alpha / 100)

            for artist in ax.collections:
                artist.set_alpha(alpha / 100)

            canvas.draw()
            canvas.flush_events()

        self.update_legend()

    def update_num_files(self):
        self.num_files_label.setText(f"{len(self.opened_files)} file(s) opened.")


# TODO Create comparator tool to compare two files?
class TEMPlotter(QMainWindow, tem_plotterUI):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setAcceptDrops(True)
        self.setWindowTitle("TEM Plotter v0.0")
        self.resize(800, 600)
        self.setWindowIcon(QtGui.QIcon(str(icons_path.joinpath('tem_plotter.png'))))
        self.err_msg = QErrorMessage()
        self.msg = QMessageBox()
        self.opened_files = []

        # X Figure
        self.x_figure = Figure()
        self.x_ax = self.x_figure.add_subplot(111)
        self.x_ax.set_ylabel('X Component Response\n()')
        self.x_ax.set_xlabel("Station")
        self.x_canvas = FigureCanvas(self.x_figure)

        toolbar = NavigationToolbar(self.x_canvas, self)

        self.x_canvas_frame.layout().addWidget(self.x_canvas)
        self.x_canvas_frame.layout().addWidget(toolbar)

        # Y Figure
        self.y_figure = Figure()
        self.y_ax = self.y_figure.add_subplot(111)
        self.y_ax.set_ylabel('Y Component Response\n()')
        self.y_ax.set_xlabel("Station")
        self.y_canvas = FigureCanvas(self.y_figure)

        toolbar = NavigationToolbar(self.y_canvas, self)

        self.y_canvas_frame.layout().addWidget(self.y_canvas)
        self.y_canvas_frame.layout().addWidget(toolbar)

        # Z Figure
        self.z_figure = Figure()
        self.z_ax = self.z_figure.add_subplot(111)
        self.z_ax.set_ylabel('Z Component Response\n()')
        self.z_ax.set_xlabel("Station")
        self.z_canvas = FigureCanvas(self.z_figure)

        toolbar = NavigationToolbar(self.z_canvas, self)

        self.z_canvas_frame.layout().addWidget(self.z_canvas)
        self.z_canvas_frame.layout().addWidget(toolbar)

        self.axes = [self.x_ax, self.y_ax, self.z_ax]
        self.canvases = [self.x_canvas, self.y_canvas, self.z_canvas]

        # Status bar
        self.num_files_label = QLabel()

        self.title = QLineEdit()
        self.title_box = QFrame()
        self.title_box.setLayout(QHBoxLayout())
        self.title_box.layout().setContentsMargins(0, 0, 0, 0)
        self.title_box.layout().addWidget(QLabel("Plot Title:"))
        self.title_box.layout().addWidget(self.title)

        # self.legend_box = QFrame()
        # self.legend_box.setLayout(QHBoxLayout())
        # self.legend_box.layout().setContentsMargins(0, 0, 0, 0)
        # self.color_by_line_cbox = QCheckBox("Color by Line")
        # self.color_by_channel_cbox = QCheckBox("Color by Channel")
        # self.legend_color_group = QButtonGroup()
        # self.legend_color_group.addButton(self.color_by_line_cbox)
        # self.legend_color_group.addButton(self.color_by_channel_cbox)
        # self.legend_box.layout().addWidget(self.color_by_line_cbox)
        # self.legend_box.layout().addWidget(self.color_by_channel_cbox)
        # self.color_by_line_cbox.setChecked(True)

        self.statusBar().addPermanentWidget(self.num_files_label, 1)
        self.statusBar().addPermanentWidget(self.title_box)

        # Signals
        self.actionOpen.triggered.connect(self.open_file_dialog)
        self.actionPrint_to_PDF.triggered.connect(self.print_pdf)
        self.actionPlot_Legend.triggered.connect(self.update_legend)

        # def replot():
        #     for ind in range(self.file_tab_widget.count()):
        #         tab = self.file_tab_widget.widget(ind).widget()
        #         self.plot_tab(tab)
        #
        # self.legend_color_group.buttonClicked.connect(replot)

        def update_title():
            """Change the title of the plots"""
            title = self.title.text()
            for ax, canvas in zip(self.axes, self.canvases):
                ax.set_title(title)

                canvas.draw()
                canvas.flush_events()

        self.title.editingFinished.connect(update_title)
        self.file_tab_widget.tabCloseRequested.connect(self.remove_tab)

        self.update_num_files()

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Space:
            for ax, canvas in zip(self.axes, self.canvases):
                ax.relim()
                ax.autoscale()

                canvas.draw()
                canvas.flush_events()

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
        for url in urls:
            self.open(url)

    def print_pdf(self, filepath=None, start_file=True):
        """Resize the figure to 11 x 8.5" and save to a PDF file"""

        if not any([self.x_ax.lines, self.y_ax.lines, self.z_ax.lines]):
            self.statusBar().showMessage(f"The plots are empty.", 1500)
            print(f"The plots are empty.")
            return

        if filepath is None:
            filepath, ext = QFileDialog.getSaveFileName(self, 'Save PDF', '', "PDF Files (*.PDF);;All Files (*.*)")

        if filepath:
            with PdfPages(filepath) as pdf:
                # Print every figure as a PDF page
                for figure in [self.x_figure, self.y_figure, self.z_figure]:

                    # Only print the figure if there are plotted lines
                    if figure.axes[0].lines:
                        old_size = figure.get_size_inches().copy()
                        figure.set_size_inches((11, 8.5))
                        pdf.savefig(figure, orientation='landscape')
                        figure.set_size_inches(old_size)

            self.statusBar().showMessage(f"PDF saved to {filepath}.", 1500)
            if start_file is True:
                os.startfile(filepath)

    def open_file_dialog(self):
        """Open files through the file dialog"""
        filepaths, ext = QFileDialog.getOpenFileNames(None, "Open FEM File", "",
                                                      "Maxwell TEM Files (*.FEM);;"
                                                      "PLATEF DAT Files (*.DAT);;"
                                                      "MUN DAT Files (*.DAT);;"
                                                      "All Files (*.*)")

        if filepaths:
            for file in filepaths:
                self.open(file)

    def open(self, filepath):
        """
        Read and plot a FEM file.
        :param filepath: str or Path object
        """
        filepath = Path(filepath)
        ext = filepath.suffix.lower()

        if ext not in ['.tem', '.dat']:
            self.msg.showMessage(self, 'Error', f"{ext[1:]} is not an implemented file extension.")
            print(f"{ext} is not supported.")
            return

        elif filepath in self.opened_files:
            print(f"{filepath.name} is already opened.")
            return

        print(f"Opening {filepath.name}.")

        # try:
        #     color = str(next(quant_colors))  # Cycles through colors
        # except StopIteration:
        #     print(f"Resetting color iterator.")
        #     quant_colors.reset()
        #     color = str(next(quant_colors))

        # Create a dict for which axes components get plotted on
        axes = {'X': self.x_ax, 'Y': self.y_ax, 'Z': self.z_ax}

        # Create a new tab and add it to the widget
        if ext == '.tem':
            tab = TEMTab(parent=self, axes=axes)
        elif ext == '.dat':
            first_line = open(filepath).readlines()[0]
            if 'Data type:' in first_line:
                components = ("X", "Y", "Z")
                component, ok_pressed = QInputDialog.getItem(self, "Choose Component", "Component:", components, 0,
                                                             False)
                if ok_pressed and component:
                    tab = MUNTab(parent=self, axes=axes, component=component)
                else:
                    return
            else:
                tab = PlateFTab(parent=self, axes=axes)
        else:
            self.msg.showMessage(self, "Error", f"{ext} is not supported.")
            return

        try:
            tab.read(filepath)
        except Exception as e:
            self.err_msg.showMessage(str(e))
            return

        # Connect signals
        tab.plot_changed_sig.connect(self.update_legend)  # Update the legend when the plot is toggled
        tab.plot_changed_sig.connect(self.update_ax_scales)  # Re-scale the plots

        # Create a new tab and add a scroll area to it, where the file tab is added to
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(250)
        scroll.setWidget(tab)
        self.file_tab_widget.addTab(scroll, filepath.name)

        self.plot_tab(tab)

        self.opened_files.append(filepath)
        self.update_num_files()

    def plot_tab(self, tab):
        # Find the tab when an index is passed (when re-plotting)
        if isinstance(tab, int):
            ind = tab
            tab = self.file_tab_widget.widget(ind).widget()

        tab.plot()

        # Add the Y axis label
        for canvas, ax in zip(self.canvases, self.axes):
            label = re.sub(r"\(.*\)", f"({tab.file.units})", ax.get_ylabel())
            if not ax.get_ylabel() or self.file_tab_widget.count() == 1:
                ax.set_ylabel(label)
            else:
                if ax.get_ylabel() != label:
                    print(f"Warning: The units for {tab.file.filepath.name} are different then the prior units.")
                    self.msg.warning(self, "Warning", f"The units for {tab.file.filepath.name} are"
                    f" different then the prior units.")

        self.update_legend()

    def remove_tab(self, ind):
        """Remove a tab"""
        # Find the tab when an index is passed (when a tab is closed)
        tab = self.file_tab_widget.widget(ind).widget()
        tab.clear()
        self.opened_files.pop(ind)
        self.file_tab_widget.removeTab(ind)

        self.update_legend()
        self.update_num_files()

    def update_legend(self):
        """Update the legend to be in alphabetical order"""

        for canvas, ax in zip(self.canvases, self.axes):
            if self.actionPlot_Legend.isChecked():
                # Only sort if there are tabs, otherwise it crashes.
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    # sort both labels and handles by labels
                    labels, handles = zip(*natsorted(zip(labels, handles), key=lambda t: t[0]))
                    ax.legend(handles, labels).set_draggable(True)
                else:
                    ax.legend().set_draggable(True)
            else:
                legend = ax.get_legend()
                if legend:
                    legend.remove()

            canvas.draw()
            canvas.flush_events()

    def update_ax_scales(self):
        """Auto re-scale every plot"""
        for ax in self.axes:
            ax.relim()
            ax.autoscale()

        for canvas in self.canvases:
            canvas.draw()
            canvas.flush_events()

    def update_num_files(self):
        self.num_files_label.setText(f"{len(self.opened_files)} file(s) opened.")


class TestRunner(QMainWindow, test_runnerUI):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setAcceptDrops(True)
        self.setWindowTitle("Test Runner v0.0")
        self.resize(800, 600)
        self.setWindowIcon(QtGui.QIcon(str(icons_path.joinpath('tem_plotter.png'))))
        self.err_msg = QErrorMessage()
        self.msg = QMessageBox()

        self.opened_files = []
        self.color_pickers = []
        self.units = ''
        self.footnote = ''

        self.header_labels = ['Folder', 'File Type', 'Data Scaling', 'Station Shift', 'Channel Start', 'Channel End',
                              'Color', 'Alpha', 'Files Found', 'Remove']
        self.table.setColumnCount(len(self.header_labels))
        self.table.setHorizontalHeaderLabels(self.header_labels)
        # Set the first column to stretch
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)

        # Figures
        self.figure, self.ax = plt.subplots()
        # self.ax2 = self.ax.twinx()  # second axes that shares the same x-axis for decay plots
        # self.ax2.get_shared_x_axes().join(self.ax, self.ax2)
        # self.ax2.set_yscale('symlog', subs=list(np.arange(2, 10, 1)))
        # self.ax2.tick_params(axis='y', which='major', labelcolor='tab:red')
        # self.ax2.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
        self.figure.set_size_inches((11, 8.5))

        def change_pdf_path():
            filepath, ext = QFileDialog.getSaveFileName(self, 'Save PDF', '', "PDF Files (*.PDF)")
            self.output_filepath_edit.setText(str(Path(filepath).with_suffix(".PDF")))

        # Signals
        self.actionConvert_IRAP_File.triggered.connect(self.open_irap_converter)
        self.actionConvert_MUN_File.triggered.connect(self.open_mun_converter)
        self.add_folder_btn.clicked.connect(self.add_row)
        self.change_pdf_path_btn.clicked.connect(change_pdf_path)
        self.table.cellClicked.connect(self.cell_clicked)
        self.include_edit.editingFinished.connect(self.filter_files)
        self.print_pdf_btn.clicked.connect(self.print_pdf)

    def cell_clicked(self, row, col):
        print(f"Row {row}, column {col} clicked.")

        if col == self.header_labels.index('Remove'):
            print(f"Removing row {row}.")
            self.table.removeRow(row)
            self.opened_files.pop(row)
            self.color_pickers.pop(row)

    def open_irap_converter(self):
        """
        Convert an IRAP File (.txt) to a .csv file for each model inside. Saves to the same directory.
        """
        default_path = str(Path(__file__).parents[1].joinpath('sample_files'))
        dlg = QFileDialog()
        irap_file, ext = dlg.getOpenFileName(self, "Select IRAP File", default_path, "IRAP Files (*.txt)")

        if irap_file:
            parser = IRAPFile()
            parser.convert(irap_file)

    def open_mun_converter(self):
        """
        Convert the files in a MUN folder. Saves to the same directory.
        """
        default_path = str(Path(__file__).parents[1].joinpath('sample_files'))
        dlg = QFileDialog()
        mun_folderpath = dlg.getExistingDirectory(self, "Select MUN Folder", default_path)

        if mun_folderpath:
            mun_file = MUNFile()
            mun_file.convert(mun_folderpath)

    def add_row(self, folderpath=None, file_type=None):
        """Add a row to the table"""
        # File type options with extensions
        # options = {"Maxwell": "*.TEM", "MUN": "*.DAT", "IRAP": "*.DAT", "PLATE": "*.DAT"}
        # colors = {"Maxwell": '#0000FF', "MUN": '#00FF00', "IRAP": "#000000", "PLATE": '#FF0000'}

        # Don't include filetypes that are already selected
        existing_filetypes = [self.table.item(row, self.header_labels.index('File Type')).text()
                              for row in range(self.table.rowCount())]
        for type in existing_filetypes:
            print(f"{type} already opened, removing from options.")
            del options[type]
            print(f"New options: {options}")

        # Don't add any  more rows if all file types have been selected
        if len(options) == 0:
            self.msg.information(self, "Maximum File Types Reached",
                                 "The maximum number of file types has been reached.")
            return

        if not folderpath:
            folderpath = QFileDialog().getExistingDirectory(self, "Select Folder", "", QFileDialog.DontUseNativeDialog)

            if not folderpath:
                print(f"No folder chosen.")
                return

        if Path(folderpath).is_dir():
            # Prompt a file type if none is given
            if file_type is None:
                file_type, ok_pressed = QInputDialog.getItem(self, "Select File Type", "File Type:",
                                                             options.keys(), 0, False)
                if not ok_pressed:
                    return

            row = self.table.rowCount()
            self.table.insertRow(row)

            # Create default items for each column
            path_item = QTableWidgetItem(str(folderpath))
            path_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            file_type_item = QTableWidgetItem(file_type)
            file_type_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            data_scaling = QTableWidgetItem("1.0")
            station_shift = QTableWidgetItem("0")
            start_ch = QTableWidgetItem("1")
            end_ch = QTableWidgetItem("99")
            color_picker = ColorButton(color=colors[file_type])
            self.color_pickers.append(color_picker)
            alpha = QTableWidgetItem("1.0")

            # Fill the row information
            for col, item in enumerate([path_item, file_type_item, data_scaling, station_shift, start_ch, end_ch,
                                        color_picker, alpha]):
                if item == color_picker:
                    self.table.setCellWidget(row, col, color_picker)
                else:
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    self.table.setItem(row, col, item)

            # Add the remove button
            """ To have an icon in the center of the cell, need to create a label and place it in a widget and 
            center the layout of the widget."""
            remove_btn_widget = QWidget()
            remove_btn_widget.setLayout(QHBoxLayout())

            remove_btn = QLabel()
            remove_btn.setMaximumSize(QtCore.QSize(16, 16))
            remove_btn.setScaledContents(True)
            remove_btn.setPixmap(QtGui.QPixmap(str(icons_path.joinpath('remove.png'))))

            remove_btn_widget.layout().setContentsMargins(0, 0, 0, 0)
            remove_btn_widget.layout().setAlignment(QtCore.Qt.AlignHCenter)
            remove_btn_widget.layout().addWidget(remove_btn)

            self.table.setCellWidget(row, self.header_labels.index('Remove'), remove_btn_widget)
            self.filter_files()
        else:
            self.msg.information(self, "Error", f"{folderpath} does not exist.")
            return

    def filter_files(self):
        """ Filter the list of files is there is a filter in place"""
        print(f"Filtering files.")
        self.opened_files = []
        options = {"Maxwell": "*.TEM", "MUN": "*.DAT", "IRAP": "*.DAT", "PLATE": "*.DAT"}
        folderpath_col = self.header_labels.index('Folder')
        file_type_col = self.header_labels.index('File Type')
        files_found_col = self.header_labels.index('Files Found')

        for row in range(self.table.rowCount()):
            # Find all the files
            file_type = self.table.item(row, file_type_col).text()
            ext = options[file_type]
            files = os_sorted(list(Path(self.table.item(row, folderpath_col).text()).glob(ext)))

            # Filter the files
            if self.include_edit.text():
                files = [f for f in files if all(
                    [string.strip() in str(f.stem) for string in self.include_edit.text().split(",")])
                         ]

            # Update number of files found in the table
            files_found_item = QTableWidgetItem(str(len(files)))
            files_found_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row, files_found_col, files_found_item)

            self.opened_files.append(files)

    def match_files(self):
        """Filter the files from each file type so only common filenames remain"""

        def get_stem(filepath):
            if filepath is None:
                return ""
            else:
                return Path(filepath).stem.upper()

        print(F"Matching files.")
        # Find which stems are common in each list
        df = pd.DataFrame(self.opened_files).T
        df_stems = df.applymap(get_stem).to_numpy()
        unique_stems = np.unique(df_stems)
        common_stems = []
        # Save the name of files that aren't in being plotted
        with open(log_file, "a+") as file:
            opened_file_types = [self.table.item(row, self.header_labels.index("File Type")).text() for row in
                                 range(self.table.rowCount())]
            for stem in unique_stems:
                if not stem:
                    continue

                if all([stem in lst for lst in df_stems.T]):
                    print(f"{stem} is in all the lists.")
                    common_stems.append(stem)
                else:
                    culprits = []  # Only used to find out which files are available for which filetypes.
                    for ind, lst in enumerate([lst for lst in df_stems.T]):
                        if stem not in lst:
                            culprits.append(opened_file_types[ind])
                    file.write(f"{stem} is not available for {', '.join(culprits)}.\n")
                    print(f"{stem} is not in all the lists.")
            file.write(">>Matching Complete<<\n\n")
        # Only keep filepaths whose stems are in the common_stems list
        filereted_files = []
        for lst in self.opened_files:
            filtered_lst = []
            for filepath in lst:
                if Path(filepath).stem.upper() in common_stems:
                    filtered_lst.append(filepath)
            filereted_files.append(filtered_lst)

        return filereted_files

    def get_plotting_info(self, file_type):
        """Return the plotting information for a file type"""
        # Find which row the file_type is on
        existing_filetypes = [self.table.item(row, self.header_labels.index('File Type')).text()
                              for row in range(self.table.rowCount())]
        row = existing_filetypes.index(file_type)

        result = dict()
        result['scaling'] = float(self.table.item(row, self.header_labels.index('Data Scaling')).text())
        result['station_shift'] = float(self.table.item(row, self.header_labels.index('Station Shift')).text())
        result['ch_start'] = int(float(self.table.item(row, self.header_labels.index('Channel Start')).text()))
        result['ch_end'] = int(float(self.table.item(row, self.header_labels.index('Channel End')).text()))
        result['color'] = self.color_pickers[row].color()
        # result['color'] = self.table.item(row, self.header_labels.index('Color')).color()  # Doesn't work???
        result['alpha'] = float(self.table.item(row, self.header_labels.index('Alpha')).text())
        return result

    def print_profiles(self, num_files_found, plotting_files, pdf_filepath):
        """
        Print the data in the files as profiles.
        :param num_files_found: Int
        :param plotting_files: dict
        :param pdf_filepath: str
        """

        def plot_maxwell(filepath, component):
            """
            Plot a Maxwell TEM file
            :param filepath: Path object
            :param component: Str, either X, Y, or Z.
            """
            parser = TEMFile()
            file = parser.parse(filepath)

            print(f"Plotting {filepath.name}.")
            properties = self.get_plotting_info('Maxwell')  # Plotting properties
            color = properties["color"]
            if not self.units:
                self.units = file.units
            else:
                if file.units != self.units:
                    self.msg.warning(self, "Different Units", f"The units of {file.filepath.name} are different then"
                    f"the existing units ({file.units} vs {self.units})")

            comp_data = file.data[file.data.COMPONENT == component]
            if comp_data.empty:
                print(f"No {component} data in {file.filepath.name}.")
                return

            channels = [f'CH{num}' for num in range(1, len(file.ch_times) + 1)]
            min_ch = properties['ch_start'] - 1
            max_ch = min(properties['ch_end'] - 1, len(channels) - 1)
            plotting_channels = channels[min_ch: max_ch + 1]

            for ind, ch in enumerate(plotting_channels):
                if ind == 0:
                    label = f"{file.filepath.name.upper()} (Maxwell)"

                    if min_ch == max_ch:
                        self.footnote += f"Maxwell file plotting channel {min_ch + 1} ({file.ch_times[max_ch]:.3f}ms).  "
                    else:
                        self.footnote += f"Maxwell file plotting channels {min_ch + 1}-{max_ch + 1}" \
                            f" ({file.ch_times[min_ch]:.3f}ms-{file.ch_times[max_ch]:.3f}ms).  "
                else:
                    label = None

                x = comp_data.STATION.astype(float) + properties['station_shift']
                y = comp_data.loc[:, ch].astype(float) * properties['scaling']

                # style = '--' if 'Q' in freq else '-'
                self.ax.plot(x, y,
                             color=color,
                             alpha=properties['alpha'],
                             label=label,
                             zorder=1)

        def plot_plate(filepath, component):
            parser = PlateFFile()
            file = parser.parse(filepath)

            print(f"Plotting {filepath.name}.")
            properties = self.get_plotting_info('PLATE')  # Plotting properties
            color = properties["color"]
            if not self.units:
                self.units = file.units
            else:
                if file.units != self.units:
                    self.msg.warning(self, "Different Units", f"The units of {file.filepath.name} are different then"
                    f"the existing units ({file.units} vs {self.units})")

            channels = [f'{num}' for num in range(1, len(file.ch_times) + 1)]
            min_ch = properties['ch_start'] - 1
            max_ch = min(properties['ch_end'] - 1, len(channels) - 1)
            plotting_channels = channels[min_ch: max_ch + 1]

            comp_data = file.data[file.data.Component == component]

            if comp_data.empty:
                print(f"No {component} data in {file.filepath.name}.")
                return

            for ind, ch in enumerate(plotting_channels):
                if ind == 0:
                    label = f"{file.filepath.name.upper()} (PLATE)"

                    if min_ch == max_ch:
                        self.footnote += f"PLATE file plotting channel {min_ch + 1} " \
                            f"({file.ch_times.loc[min_ch] * 1000:.3f}ms).  "
                    else:
                        self.footnote += f"PLATE file plotting channels {min_ch + 1}-{max_ch + 1}" \
                            f" ({file.ch_times.loc[min_ch] * 1000:.3f}ms-{file.ch_times.loc[max_ch] * 1000:.3f}ms).  "
                else:
                    label = None

                x = comp_data.Station.astype(float) + properties['station_shift']
                y = comp_data.loc[:, ch].astype(float) * properties['scaling']

                self.ax.plot(x, y,
                             color=color,
                             alpha=properties['alpha'],
                             # lw=count / 100,
                             label=label,
                             zorder=2)

        def plot_mun(filepath, component):
            parser = MUNFile()
            file = parser.parse(filepath)

            print(f"Plotting {filepath.name}.")
            properties = self.get_plotting_info('MUN')  # Plotting properties
            color = properties["color"]
            if not self.units:
                self.units = file.units
            else:
                if file.units != self.units:
                    self.msg.warning(self, "Different Units", f"The units of {file.filepath.name} are different then "
                                                              f"the existing units ({file.units} vs {self.units})")

            channels = [f'CH{num}' for num in range(1, len(file.ch_times) + 1)]
            min_ch = properties['ch_start'] - 1
            max_ch = min(properties['ch_end'] - 1, len(channels) - 1)
            plotting_channels = channels[min_ch: max_ch + 1]

            comp_data = file.data[file.data.Component == component]

            if comp_data.empty:
                print(f"No {component} data in {file.filepath.name}.")
                return

            for ind, ch in enumerate(plotting_channels):
                # If coloring by channel, uses the rainbow color iterator and the label is the channel number.
                if ind == 0:
                    label = f"{file.filepath.name.upper()} (MUN)"

                    if min_ch == max_ch:
                        self.footnote += f"MUN file plotting channel {min_ch + 1} ({file.ch_times[max_ch]:.3f}ms).  "
                    else:
                        self.footnote += f"MUN file plotting channels {min_ch + 1}-{max_ch + 1}" \
                                         f" ({file.ch_times[min_ch]:.3f}ms-{file.ch_times[max_ch]:.3f}ms).  "
                else:
                    label = None

                x = comp_data.Station.astype(float) + properties['station_shift']
                y = comp_data.loc[:, ch].astype(float) * properties['scaling']

                self.ax.plot(x, y,
                             color=color,
                             alpha=properties['alpha'],
                             label=label,
                             zorder=3)

        def plot_irap(filepath, component):
            """
            Plot an IRAP DAT file
            :param filepath: Path object
            :param component: Str, either X, Y, or Z.
            """
            parser = IRAPFile()
            file = parser.parse(filepath)

            print(f"Plotting {filepath.name}.")
            properties = self.get_plotting_info('IRAP')  # Plotting properties
            color = properties["color"]
            # Units are not in IRAP's files
            # if not self.units:
            #     self.units = file.units
            # else:
            #     if file.units != self.units:
            #         self.msg.warning(self, "Different Units", f"The units of {file.filepath.name} are different then"
            #                                                   f"the existing units ({file.units} vs {self.units})")

            comp_data = file.data[file.data.Component == component]
            if comp_data.empty:
                print(f"No {component} data in {file.filepath.name}.")
                return

            channels = file.ch_times.index
            min_ch = properties['ch_start'] - 1
            max_ch = min(properties['ch_end'] - 1, len(channels) - 1)
            plotting_channels = channels[min_ch: max_ch + 1]

            for ind, ch in enumerate(plotting_channels):
                if ind == 0:
                    label = f"{file.filepath.name.upper()} (IRAP)"

                    min_time, max_time = file.ch_times.loc[min_ch, "Start"], file.ch_times.loc[max_ch, "End"]
                    if min_ch == max_ch:
                        self.footnote += f"IRAP file plotting channel {min_ch + 1} ({min_time:.3f}ms).  "
                    else:
                        self.footnote += f"IRAP file plotting channels {min_ch + 1}-{max_ch + 1}" \
                                         f" ({min_time:.3f}ms-{max_time:.3f}ms).  "
                else:
                    label = None

                x = comp_data.Station.astype(float) + properties['station_shift']
                y = comp_data.loc[:, ch].astype(float) * properties['scaling']

                self.ax.plot(x, y,
                             color=color,
                             alpha=properties['alpha'],
                             label=label,
                             zorder=1)

        def get_fixed_range():
            """Find the Y range of each file"""
            progress.setLabelText("Calculating Ranges")
            max_parser = TEMFile()
            plate_parser = PlateFFile()
            mun_parser = MUNFile()
            count = 0

            mins, maxs = [], []
            for max_filepath in plotting_files["Maxwell"]:
                if progress.wasCanceled():
                    break

                max_file = max_parser.parse(max_filepath)
                rng = max_file.get_range()
                mins.append(rng[0] * self.get_plotting_info('Maxwell')["scaling"])
                maxs.append(rng[1] * self.get_plotting_info('Maxwell')["scaling"])

                count += 1
                progress.setValue(count)

            for plate_filepath in plotting_files["PLATE"]:
                if progress.wasCanceled():
                    break

                plate_file = plate_parser.parse(plate_filepath)
                rng = plate_file.get_range()
                mins.append(rng[0] * self.get_plotting_info('PLATE')["scaling"])
                maxs.append(rng[1] * self.get_plotting_info('PLATE')["scaling"])

                count += 1
                progress.setValue(count)

            return min(mins), max(maxs)

        def format_figure(component):
            if self.log_y_cbox.isChecked():
                if self.plot_profiles_rbtn.isChecked():
                    self.ax.set_yscale('symlog',
                                       linthresh=10,
                                       linscale=1. / math.log(10),
                                       subs=list(np.arange(2, 10, 1)))
                else:
                    self.ax.set_yscale('symlog', subs=list(np.arange(2, 10, 1)))
            else:
                self.ax.set_yscale('linear')

            # Set the labels
            self.ax.set_xlabel(f"Station")
            self.ax.set_ylabel(f"{component} Component Response\n({self.units})")
            self.ax.set_title(self.test_name_edit.text())

            if self.custom_stations_cbox.isChecked():
                self.ax.set_xlim([self.station_start_sbox.value(), self.station_end_sbox.value()])
            if self.fixed_range_cbox.isChecked():
                y_range = np.array(get_fixed_range())
                self.ax.set_ylim([y_range[0], y_range[1]])

            # Create the legend
            handles, labels = self.ax.get_legend_handles_labels()

            if handles:
                # sort both labels and handles by labels
                labels, handles = zip(*os_sorted(zip(labels, handles), key=lambda t: t[0]))
                self.ax.legend(handles, labels).set_draggable(True)

            # Add the footnote
            self.ax.text(0.995, 0.01, self.footnote,
                         ha='right',
                         va='bottom',
                         size=6,
                         transform=self.figure.transFigure)

        progress = QProgressDialog("Processing...", "Cancel", 0, int(num_files_found))
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setWindowTitle("Printing Profiles")
        progress.show()

        count = 0
        progress.setValue(count)
        progress.setLabelText("Printing Profile Plots")
        with PdfPages(pdf_filepath) as pdf:
            for maxwell_file, mun_file, irap_file, plate_file in list(zip_longest(*plotting_files.values(),
                                                                                  fillvalue=None))[:]:
                if progress.wasCanceled():
                    print(f"Process cancelled.")
                    break

                print(f"Plotting set {count + 1}/{int(num_files_found)}")
                for component in [cbox.text() for cbox in [self.x_cbox, self.y_cbox, self.z_cbox] if cbox.isChecked()]:
                    self.footnote = ''

                    # Plot the files
                    if maxwell_file:
                        plot_maxwell(maxwell_file, component)
                    if mun_file:
                        plot_mun(mun_file, component)
                    if irap_file:
                        plot_irap(irap_file, component)
                    if plate_file:
                        plot_plate(plate_file, component)

                    format_figure(component)
                    pdf.savefig(self.figure, orientation='landscape')
                    self.ax.clear()

                count += 1
                progress.setValue(count)

        # os.startfile(pdf_filepath)

    def print_decays(self, num_files_found, plotting_files, pdf_filepath):
        """
        Plot the decays of stations, based on programmed criteria.
        """

        def plot_maxwell(filepath, component):
            """
            Plot a Maxwell TEM file
            :param filepath: Path object
            :param component: Str, either X, Y, or Z.
            """
            parser = TEMFile()
            file = parser.parse(filepath)

            print(f"Plotting {filepath.name}.")
            properties = self.get_plotting_info('Maxwell')  # Plotting properties
            color = properties["color"]
            if not self.units:
                self.units = file.units
            else:
                if file.units != self.units:
                    self.msg.warning(self, "Different Units", f"The units of {file.filepath.name} are different then"
                    f"the existing units ({file.units} vs {self.units})")

            comp_data = file.data[file.data.COMPONENT == component]
            if comp_data.empty:
                print(f"No {component} data in {file.filepath.name}.")
                return

            channels = [f'CH{num}' for num in range(1, len(file.ch_times) + 1)]
            min_ch = properties['ch_start'] - 1
            max_ch = min(properties['ch_end'] - 1, len(channels) - 1)
            plotting_channels = channels[min_ch: max_ch + 1]

            """Plotting decay for run-on effects"""

            data = comp_data.loc[:, plotting_channels]
            data.index = comp_data.STATION
            last_ch_data = data.loc[:, plotting_channels[-1]]

            # Find the station where the response is highest
            station = last_ch_data.idxmax()
            print(f"Plotting station {station}.")

            x = file.ch_times[min_ch: max_ch + 1]
            decay = data.loc[station, plotting_channels] * properties['scaling']

            label = f"{file.filepath.name.upper()} (Maxwell)"

            self.footnote += f"Maxwell file plotting station {station}.  "

            # style = '--' if 'Q' in freq else '-'
            self.ax.plot(x, decay,
                         color=color,
                         alpha=properties['alpha'],
                         label=label,
                         zorder=1)

            # self.ax2.plot(x, decay,
            #               color='tab:red',
            #               alpha=properties['alpha'],
            #               label="Logarithmic-scale",
            #               zorder=1)

        def plot_plate(filepath, component):
            raise NotImplementedError("PLATE decay plots not implemented yet.")

        def plot_mun(filepath, component):
            raise NotImplementedError("MUN decay plots not implemented yet.")

        def plot_irap(filepath, component):
            raise NotImplementedError("IRAP decay plots not implemented yet.")

        def format_figure(component):
            # Set the labels
            self.ax.set_xlabel(f"Time (ms)")
            self.ax.set_ylabel(f"{component} Component Response\n({self.units})")
            # self.ax.set_title(f"{self.test_name_edit.text()} - {maxwell_file.stem}")

            if self.custom_stations_cbox.isChecked():
                self.ax.set_xlim([self.station_start_sbox.value(), self.station_end_sbox.value()])

            # Create the legend
            handles, labels = self.ax.get_legend_handles_labels()
            # handles2, labels2 = self.ax2.get_legend_handles_labels()
            # handles.extend(handles2)
            # labels.extend(labels2)

            if handles:
                # sort both labels and handles by labels
                labels, handles = zip(*os_sorted(zip(labels, handles), key=lambda t: t[0]))
                self.ax.legend(handles, labels).set_draggable(True)

            # Add the footnote
            self.ax.text(0.995, 0.01, self.footnote,
                         ha='right',
                         va='bottom',
                         size=6,
                         transform=self.figure.transFigure)

        # self.ax2.get_yaxis().set_visible(True)
        self.ax.tick_params(axis='y', labelcolor='blue')
        self.ax.set_yscale('linear')
        progress = QProgressDialog("Processing...", "Cancel", 0, int(num_files_found))
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setWindowTitle("Printing Decays")
        progress.show()
        count = 0

        with PdfPages(pdf_filepath) as pdf:
            for maxwell_file, mun_file, irap_file, plate_file in list(zip_longest(*plotting_files.values(),
                                                                                   fillvalue=None))[:]:
                if progress.wasCanceled():
                    print(f"Process cancelled.")
                    break

                print(f"Plotting set {count + 1}/{int(num_files_found)}")
                for component in [cbox.text() for cbox in [self.x_cbox, self.y_cbox, self.z_cbox] if cbox.isChecked()]:
                    self.footnote = ''

                    # Plot the files
                    if maxwell_file:

                        def is_eligible(file):
                            comp_data = file.data[file.data.COMPONENT == component]
                            if comp_data.empty:
                                print(f"No {component} data in {file.filepath.name}.")
                                return False

                            properties = self.get_plotting_info('Maxwell')
                            channels = [f'CH{num}' for num in range(1, len(file.ch_times) + 1)]
                            min_ch = properties['ch_start'] - 1
                            max_ch = min(properties['ch_end'] - 1, len(channels) - 1)
                            # min_ch = 21 - 1
                            # max_ch = 44 - 1
                            plotting_channels = channels[min_ch: max_ch + 1]
                            data = comp_data.loc[:, plotting_channels]
                            data.index = comp_data.STATION
                            """Plotting decay for run-on effects"""
                            last_ch_data = data.loc[:, plotting_channels[-1]] * properties['scaling']
                            if last_ch_data.abs().max() >= 5:
                                return True
                            else:
                                print(
                                    f"Skipping {file.filepath.name} because the max value in the last channel is {last_ch_data.max():.2f}.")
                                return False

                        parser = TEMFile()
                        file = parser.parse(maxwell_file)
                        if is_eligible(file):
                            plot_maxwell(maxwell_file, component)
                        else:
                            continue

                    if mun_file:
                        plot_mun(mun_file, component)
                    if irap_file:
                        plot_irap(irap_file, component)
                    if plate_file:
                        plot_plate(plate_file, component)

                    format_figure(component)
                    # plt.show()
                    pdf.savefig(self.figure, orientation='landscape')
                    self.ax.clear()
                    # self.ax2.clear()
                    # self.ax2.set_yscale('symlog', subs=list(np.arange(2, 10, 1)))
                    # self.ax2.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))

                count += 1
                progress.setValue(count)

        # os.startfile(pdf_filepath)

    def print_run_on_comparison(self, plotting_files, pdf_filepath):
        """
        Print the run-on effect calculation plots
        :param plotting_files: dict
        :param pdf_filepath: str
        """

        def plot_maxwell_decays(files, pdf):
            """
            Calculate the run-on effect for Maxwell files.
            :param files: list of filepaths of the maxwell files
            :param pdf: str, PDF file to save to.
            """
            print(f"Printing Maxwell run-on effect")
            # self.ax2.get_yaxis().set_visible(False)
            self.ax.tick_params(axis='y', labelcolor='k')

            properties = self.get_plotting_info('Maxwell')  # Plotting properties
            color = properties["color"]

            progress = QProgressDialog("Parsing TEM files", "Cancel", 0, len(files))
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setWindowTitle("Printing Maxwell run-on")
            progress.show()
            count = 0

            # Gather all the TEM files in the folder
            tem_files = []
            for file in files:
                progress.setLabelText(f"Parsing {Path(file).name}")
                tem_file = TEMFile()
                tem_file.parse(file)
                tem_files.append(tem_file)

                count += 1
                progress.setValue(count)
            base_file = tem_files[0]  # Use the first file as a base file for determining which station to plot

            channels = [f'CH{num}' for num in range(1, len(base_file.ch_times) + 1)]
            min_ch = properties['ch_start'] - 1
            max_ch = min(properties['ch_end'] - 1, len(channels) - 1)
            plotting_channels = channels[min_ch: max_ch + 1]

            count = 0
            progress.setValue(count)
            progress.setMaximum(3)

            for component in [cbox.text() for cbox in [self.x_cbox, self.y_cbox, self.z_cbox] if cbox.isChecked()]:
                if progress.wasCanceled():
                    print(f"Process cancelled.")
                    return
                progress.setLabelText(f"Plotting {component} component.")
                print(f"Plotting {component} component.")

                comp_data = base_file.data[base_file.data.COMPONENT == component]
                if comp_data.empty:
                    print(f"No {component} data in {base_file.filepath.name}.")
                    return

                self.footnote = ''

                data = comp_data.loc[:, plotting_channels]
                data.index = comp_data.STATION
                last_ch_data = data.loc[:, plotting_channels[-1]]

                # Find the station where the response is highest
                station = last_ch_data.idxmax()
                self.footnote += f"Maxwell file plotting station {station}.  "
                print(f"Plotting station {station}.")

                # Create a data frame from all the data in all the files in the folder
                df = pd.DataFrame()
                for ind, tem_file in enumerate(tem_files):
                    file_comp_data = tem_file.data[tem_file.data.COMPONENT == component]
                    file_comp_data.index = file_comp_data.STATION
                    df[str(ind + 1)] = file_comp_data.loc[station, plotting_channels]
                df = df.T

                # Calculate the decay
                decay = []
                n = 9  # Number of files to complete 1 timebase
                for ch in list(range(0, len(plotting_channels))):
                    print(f"Calculating channel {ch + min_ch + 1}.")
                    # response = df.iloc[0, ch] - df.iloc[n + 1, ch] - df.iloc[2, ch] + df.iloc[n + 3, ch] + \
                    #            df.iloc[4, ch] - df.iloc[n + 5, ch] - df.iloc[6, ch] + df.iloc[n + 7, ch] + \
                    #            df.iloc[8, ch]

                    # response = F[0] - F[n + 1] - F[2] + F[n + 3] + F[4] - F[n + 5] - F[6] + F[n + 7] + F[8]

                    # On-time calculation
                    response = df.iloc[0, ch] - df.iloc[10, ch] - df.iloc[2, ch] + df.iloc[12, ch] + \
                               df.iloc[4, ch] - df.iloc[14, ch] - df.iloc[6, ch] + df.iloc[16, ch] + df.iloc[8, ch]
                    decay.append(response)

                # Include a test file for comparison
                parser = TEMFile()
                base_folder = Path(__file__).parents[1].joinpath(r'sample_files\Aspect ratio\Maxwell\2m stations')
                other_file = parser.parse(base_folder.joinpath(r'600x600C.tem'))
                other_file_data = other_file.data[other_file.data.COMPONENT == component]
                other_file_data.index = other_file_data.STATION
                other_file_decay = other_file_data.loc[station, plotting_channels] * properties['scaling']

                # Plot the data
                x = base_file.ch_times[min_ch: max_ch + 1]
                decay = np.array(decay) * properties['scaling']
                # self.ax.set_yscale('symlog', subs=list(np.arange(2, 10, 1)), linthresh=10, linscale=1. / math.log(10))
                self.ax.plot(x, decay, color=color, label="Calculated", alpha=properties['alpha'])
                self.ax.plot(x, other_file_decay, color='r', label="600x600C", alpha=0.6)

                # Set the labels
                self.ax.set_xlabel(f"Time (ms)")
                self.ax.set_ylabel(f"{component} Component Response\n({base_file.units})")
                self.ax.set_title(self.test_name_edit.text())

                # Add the footnote
                self.ax.text(0.995, 0.01, self.footnote,
                             ha='right',
                             va='bottom',
                             size=6,
                             transform=self.figure.transFigure)

                # Create the legend
                self.ax.legend()

                # Save the PDF
                pdf.savefig(self.figure, orientation='landscape')

                self.ax.clear()
                count += 1
                progress.setValue(count)

        def plot_plate(filepath, component):
            raise NotImplementedError("PLATE run-on not implemented yet.")

        def plot_mun(filepath, component):
            raise NotImplementedError("MUN run-on not implemented yet.")

        def plot_irap(filepath, component):
            raise NotImplementedError("IRAP run-on not implemented yet.")

        self.ax.set_yscale('linear')
        with PdfPages(pdf_filepath) as pdf:
            if plotting_files['Maxwell']:
                plot_maxwell_decays(plotting_files['Maxwell'], pdf)
            if plotting_files['MUN']:
                plot_mun(plotting_files['MUN'], pdf)
            if plotting_files['IRAP']:
                plot_irap(plotting_files['IRAP'], pdf)
            if plotting_files['PLATE']:
                plot_plate(plotting_files['PLATE'], pdf)

        # os.startfile(pdf_filepath)

    def print_run_on_convergence(self, plotting_files, pdf_filepath):
        """
        Print the run-on effect calculation plots
        :param plotting_files: dict
        :param pdf_filepath: str
        """

        def plot_maxwell_convergence(files, pdf):
            """
            Calculate the run-on effect half-cycle convergence Maxwell files.
            :param files: list, Path filepaths.
            :param pdf: str, PDF file to save to.
            """
            print(f"Printing Maxwell run-on convergence")
            properties = self.get_plotting_info('Maxwell')  # Plotting properties
            colors = {'X': 'r', 'Y': 'g', 'Z': 'b'}

            progress = QProgressDialog("Parsing TEM files", "Cancel", 0, 3)
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setWindowTitle("Printing Maxwell run-on")
            progress.show()
            count = 0

            for file in files:
                print(f"Plotting {file.name} ({count}/{len(files)}).")
                self.footnote = ''
                # self.ax2.get_yaxis().set_visible(False)
                self.ax.tick_params(axis='y', labelcolor='k')

                tem_file = TEMFile()
                tem_file.parse(file)

                # Find the comparison file
                base_folder = Path(__file__).parents[1].joinpath(r'sample_files\Aspect ratio\Maxwell\2m stations')
                other_file = base_folder.joinpath(file.name)
                if not other_file.is_file():
                    print(f"Cannot find {other_file}.")
                    count += 1
                    progress.setValue(count)
                    continue

                base_file = TEMFile()
                base_file = base_file.parse(other_file)

                channels = [f'CH{num}' for num in range(1, len(tem_file.ch_times) + 1)]

                progress.setValue(count)

                for component in [cbox.text() for cbox in [self.x_cbox, self.y_cbox, self.z_cbox] if cbox.isChecked()]:
                    if progress.wasCanceled():
                        print(f"Process cancelled.")
                        return
                    print(f"Plotting {component} component.")

                    comp_data = tem_file.data[tem_file.data.COMPONENT == component]
                    base_file_data = base_file.data[base_file.data.COMPONENT == component]
                    base_file_data.index = base_file_data.STATION

                    data = comp_data.loc[:, channels]
                    data.index = comp_data.STATION
                    last_ch_data = data.loc[:, channels[-1]]

                    # Find the station where the response is highest
                    station = last_ch_data.idxmax()
                    self.footnote += f"{component} component plotting station {station}.  "

                    # Create a data frame from all the data in all the files in the folder
                    file_comp_data = tem_file.data[tem_file.data.COMPONENT == component]
                    file_comp_data.index = file_comp_data.STATION
                    df = file_comp_data.loc[station, channels]
                    df = df.T
                    # n = int(float(tem_file.off_time) / 50)  # Number of sequential 50ms timebases
                    #
                    # terms = []
                    # # Build the formula
                    # for i in range(0, n):
                    #     p = i % 4
                    #     if p == 0:
                    #         terms.append(df.iloc[i])
                    #     elif p == 1:
                    #         terms.append(- df.iloc[n + i])
                    #     elif p == 2:
                    #         terms.append(- df.iloc[i])
                    #     else:
                    #         terms.append(df.iloc[n + i])
                    #
                    # # Plot the data
                    # xs = range(1, math.floor(n / 2) + 1)
                    # responses = np.array([sum(terms[:2 * n]) for n in xs]) * properties['scaling']

                    n = int(len(tem_file.ch_times) / 2)
                    terms = []
                    count = 0
                    for i in range(0, n):
                        if i % 2 == 0:
                            term = df.iloc[i] - df.iloc[n + i]
                        else:
                            term = -df.iloc[i] + df.iloc[n + i]
                        terms.append(term)

                    # Plot the data
                    xs = range(1, n + 1)
                    responses = np.array([sum(terms[:i]) for i in range(1, n + 1)]) * properties['scaling']

                    self.ax.plot(xs[:10], responses[:10],
                                 color=colors[component],
                                 alpha=properties['alpha'],
                                 label=f"{component} Component")

                    # Add the value of channel 44 from the comparisson file
                    base_file_channel_value = base_file_data.loc[station, "CH44"] * properties['scaling']
                    self.ax.plot(xs[:10], np.repeat(base_file_channel_value, len(xs[:10])),
                                 color=colors[component],
                                 ls='--',
                                 lw=1.,
                                 zorder=-1,
                                 alpha=properties['alpha'])

                # Set the labels
                self.ax.set_xlabel(f"Half-cycles")
                self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                self.ax.set_ylabel("Cumulative Sum")
                self.ax.set_title(f"{self.test_name_edit.text()} - {file.stem}")

                # Add the footnote
                self.ax.text(0.995, 0.01, self.footnote,
                             ha='right',
                             va='bottom',
                             size=6,
                             transform=self.figure.transFigure)

                # Create the legend
                self.ax.legend()

                # Save the PDF
                pdf.savefig(self.figure, orientation='landscape')

                self.ax.clear()
                count += 1
                progress.setValue(count)

        self.ax.set_yscale('linear')
        with PdfPages(pdf_filepath) as pdf:
            if plotting_files['Maxwell']:
                plot_maxwell_convergence(plotting_files['Maxwell'], pdf)

        # os.startfile(pdf_filepath)

    def tabulate_run_on_convergence(self, plotting_files):

        def tabulate_maxwell_convergence(files):
            """
            Calculate the run-on effect half-cycle convergence Maxwell files and tabulate the results.
            :param files: list, Path filepaths.
            """
            print(f"Printing Maxwell run-on convergence")
            properties = self.get_plotting_info('Maxwell')  # Plotting properties

            convergence_df = pd.DataFrame()

            progress = QProgressDialog("Parsing TEM files", "Cancel", 0, len(files))
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setWindowTitle("Printing Maxwell run-on")
            progress.show()
            count = 0

            for file in files:
                print(f"Plotting {file.name} ({count}/{len(files)}).")
                self.footnote = ''
                # self.ax2.get_yaxis().set_visible(False)
                self.ax.tick_params(axis='y', labelcolor='k')

                tem_file = TEMFile()
                tem_file.parse(file)

                # Find the comparison file
                base_folder = Path(__file__).parents[1].joinpath(r'sample_files\Aspect ratio\Maxwell\2m stations')
                other_file = base_folder.joinpath(file.name)
                if not other_file.is_file():
                    print(f"Cannot find {other_file}.")
                    count += 1
                    progress.setValue(count)
                    continue
                base_file = TEMFile()
                base_file = base_file.parse(other_file)

                channels = [f'CH{num}' for num in range(1, len(tem_file.ch_times) + 1)]

                progress.setValue(count)

                for component in [cbox.text() for cbox in [self.x_cbox, self.y_cbox, self.z_cbox] if cbox.isChecked()]:
                    if progress.wasCanceled():
                        print(f"Process cancelled.")
                        return
                    print(f"Plotting {component} component.")

                    comp_data = tem_file.data[tem_file.data.COMPONENT == component]
                    base_file_data = base_file.data[base_file.data.COMPONENT == component]
                    base_file_data.index = base_file_data.STATION

                    data = comp_data.loc[:, channels]
                    data.index = comp_data.STATION
                    last_ch_data = data.loc[:, channels[-1]]

                    # Find the station where the response is highest
                    station = last_ch_data.idxmax()
                    self.footnote += f"{component} component plotting station {station}.  "

                    # Create a data frame from all the data in all the files in the folder
                    file_comp_data = tem_file.data[tem_file.data.COMPONENT == component]
                    file_comp_data.index = file_comp_data.STATION
                    df = file_comp_data.loc[station, channels]
                    df = df.T

                    base_file_channel_value = base_file_data.loc[station, "CH44"] * properties['scaling']

                    n = int(len(tem_file.ch_times) / 2)
                    terms = []
                    for i in range(0, n):
                        if i % 2 == 0:
                            term = df.iloc[i] - df.iloc[n + i]
                        else:
                            term = -df.iloc[i] + df.iloc[n + i]
                        terms.append(term)

                    # Plot the data
                    xs = range(1, n + 1)
                    responses = np.array([sum(terms[:i]) for i in range(1, n + 1)]) * properties['scaling']

                    diff = base_file_channel_value - responses
                    convergence_df[f"{file.stem} - {component}"] = np.abs(diff)

                count += 1

            convergence_df = convergence_df.T.round(decimals=2).set_axis([str(num) for num in range(1, len(xs) + 1)],
                                                                         axis=1)

            def find_convergence(row, thresh):
                for ind, col in enumerate(row):
                    if all([k < thresh for k in row[ind:]]):
                        return ind + 1  # Guaranteed to happen at 5

            # Find the first column where all columns past it have a difference less than 1.
            convergences = []
            for i, row in convergence_df.iterrows():
                print(f"Items in row:\n{row}")
                convergence = find_convergence(row, 0.1)
                convergences.append(convergence)

            convergence_df['Required_half_cycles'] = convergences
            convergence_df.loc[:, "Required_half_cycles"].to_csv(output_filepath)
            # os.startfile(output_filepath)

        output_filepath = self.output_filepath_edit.text()

        if plotting_files['Maxwell']:
            tabulate_maxwell_convergence(plotting_files['Maxwell'])

    def print_pdf(self, from_script=False):
        """Create the PDF"""
        if self.table.rowCount() == 0:
            return

        pdf_filepath = self.output_filepath_edit.text()
        if not pdf_filepath:
            self.msg.information(self, "Error", f"PDF output path must not be empty.")
            return

        # Ensure there are equal number of files found for each file type
        num_files = []
        # num_files_found = self.table.item(0, self.header_labels.index("Files Found")).text()
        for row in range(self.table.rowCount()):
            num_files.append(self.table.item(row, self.header_labels.index("Files Found")).text())

        if not all([int(num) == int(num_files[0]) for num in num_files]):
            if from_script is False:
                response = self.msg.question(self, "Unequal Files", "A different number of files was found for each "
                                                                    "filetype. Do you wish to only plot common files?",
                                             self.msg.Yes, self.msg.No)
                if response == self.msg.Yes:
                    opened_files = self.match_files()
                else:
                    return
            else:
                opened_files = self.match_files()
        else:
            opened_files = self.opened_files.copy()

        num_files_found = len(opened_files[0])
        t0 = time.time()

        # Create a dictionary of files to plot
        plotting_files = {"Maxwell": [], "MUN": [], "IRAP": [], "PLATE": []}
        for row in range(self.table.rowCount()):
            files = os_sorted(opened_files[row])
            file_type = self.table.item(row, self.header_labels.index('File Type')).text()

            for file in files:
                plotting_files[file_type].append(file)

        if not any(plotting_files.values()):
            raise ValueError("No plotting files found.")

        if self.plot_profiles_rbtn.isChecked():
            self.print_profiles(num_files_found, plotting_files, pdf_filepath)
        elif self.plot_decays_rbtn.isChecked():
            self.print_decays(num_files_found, plotting_files, pdf_filepath)
        elif self.plot_run_on_comparison_rbtn.isChecked():
            self.print_run_on_comparison(plotting_files, pdf_filepath)
        elif self.plot_run_on_convergence_rbtn.isChecked():
            self.print_run_on_convergence(plotting_files, pdf_filepath)
        elif self.table_run_on_convergence_rbtn.isChecked():
            self.tabulate_run_on_convergence(plotting_files)

        print(f"Plotting complete after {math.floor((time.time() - t0) / 60):02.0f}:{(time.time() - t0) % 60:02.0f}")


if __name__ == '__main__':
    import time

    app = QApplication(sys.argv)

    sample_files = Path(__file__).parents[1].joinpath('sample_files')

    # fem_file = sample_files.joinpath(r'Maxwell files\FEM\Horizontal Plate 100S Normalized.fem')
    # tem_file = sample_files.joinpath(r'Aspect ratio\Maxwell\5x150A.TEM')

    def plot_max(axes, file, ch_start, ch_end, ch_step=1, name="", station_shift=0, single_file=False,
                 data_scaling=1., alpha=1., line_color=None, ls=None, x_min=None, x_max=None,
                 y_min=None, y_max=None, incl_label=True):
        x_ax, z_ax, x_ax_log, z_ax_log = axes
        rainbow_colors = cm.jet(np.linspace(0, ch_step, (ch_end - ch_start) + 1))
        x_ax.set_prop_cycle(cycler('color', rainbow_colors))
        x_ax_log.set_prop_cycle(cycler('color', rainbow_colors))
        z_ax.set_prop_cycle(cycler('color', rainbow_colors))
        z_ax_log.set_prop_cycle(cycler('color', rainbow_colors))

        x_data = file.data[file.data.COMPONENT == "X"]
        z_data = file.data[file.data.COMPONENT == "Z"]

        channels = [f'CH{num}' for num in range(1, len(file.ch_times) + 1)]
        min_ch = ch_start - 1
        max_ch = min(ch_end - 1, len(channels) - 1)
        plotting_channels = channels[min_ch: max_ch + 1: ch_step]
        if single_file is True:
            line_color = None

        for ind, ch in enumerate(plotting_channels):
            if incl_label is True:
                if single_file is True:
                    label = f"{file.ch_times[min_ch + (ind * ch_step)]:.3f}ms"
                    # label = ch
                else:
                    if ind == 0:
                        label = name
                    else:
                        label = None
            else:
                label = None

            x = z_data.STATION.astype(float) + station_shift
            zz = z_data.loc[:, ch].astype(float) * data_scaling
            xx = x_data.loc[:, ch].astype(float) * data_scaling

            for ax in [x_ax, x_ax_log]:
                ax.plot(x, xx,
                        color=line_color,
                        alpha=alpha,
                        # alpha=1 - (ind / (len(plotting_channels))) * 0.9,
                        label=label,
                        ls=ls,
                        zorder=1)
            for ax in [z_ax, z_ax_log]:
                ax.plot(x, zz,
                        color=line_color,
                        alpha=alpha,
                        # alpha=1 - (ind / (len(plotting_channels))) * 0.9,
                        label=label,
                        ls=ls,
                        zorder=1)

            for ax in axes:
                if x_min and x_max:
                    ax.set_xlim([x_min, x_max])
                else:
                    ax.set_xlim([x.min(), x.max()])
                if y_min and y_max:
                    ax.set_ylim([y_min, y_max])

    def plot_mun(axes, file, ch_start, ch_end, ch_step=1, name="", station_shift=0, data_scaling=1., alpha=1.,
                 line_color=None, ls=None, x_min=None, x_max=None, y_min=None, y_max=None, single_file=False,
                 incl_label=True, filter=False):
        x_ax, z_ax, x_ax_log, z_ax_log = axes
        rainbow_colors = cm.jet(np.linspace(0, ch_step, (ch_end - ch_start) + 1))
        x_ax.set_prop_cycle(cycler('color', rainbow_colors))
        x_ax_log.set_prop_cycle(cycler('color', rainbow_colors))
        z_ax.set_prop_cycle(cycler('color', rainbow_colors))
        z_ax_log.set_prop_cycle(cycler('color', rainbow_colors))

        x_data = file.data[file.data.Component == "X"]
        z_data = file.data[file.data.Component == "Z"]

        channels = [f'CH{num}' for num in range(1, len(file.ch_times) + 1)]
        min_ch = ch_start - 1
        max_ch = min(ch_end - 1, len(channels) - 1)
        plotting_channels = channels[min_ch: max_ch + 1: ch_step]
        if single_file is True:
            line_color = None

        for ind, ch in enumerate(plotting_channels):
            if incl_label is True:
                if single_file is True:
                    label = f"{file.ch_times[min_ch + ind]:.3f}ms"
                    # label = ch
                else:
                    if ind == 0:
                        label = name
                    else:
                        label = None
            else:
                label = None

            x = z_data.Station.astype(float) + station_shift
            zz = z_data.loc[:, ch].astype(float) * data_scaling  # * -1
            xx = x_data.loc[:, ch].astype(float) * data_scaling  # * -1

            if filter is True:
                zz = savgol_filter(zz, 21, 3)
                xx = savgol_filter(xx, 21, 3)

            for ax in [x_ax, x_ax_log]:
                ax.plot(x, xx,
                        color=line_color,
                        alpha=alpha,
                        # alpha=1 - (ind / (len(plotting_channels))) * 0.9,
                        label=label,
                        ls=ls,
                        zorder=1)
            for ax in [z_ax, z_ax_log]:
                ax.plot(x, zz,
                        color=line_color,
                        alpha=alpha,
                        # alpha=1 - (ind / (len(plotting_channels))) * 0.9,
                        label=label,
                        ls=ls,
                        zorder=1)

            for ax in axes:
                if x_min and x_max:
                    ax.set_xlim([x_min, x_max])
                else:
                    ax.set_xlim([x.min(), x.max()])
                if y_min and y_max:
                    ax.set_ylim([y_min, y_max])

    def plot_plate(axes, file, ch_start, ch_end, ch_step=1, name="", station_shift=0, data_scaling=1., alpha=1.,
                 line_color=None, ls=None, x_min=None, x_max=None, y_min=None, y_max=None, single_file=False,
                 incl_label=True, filter=False):
        x_ax, z_ax, x_ax_log, z_ax_log = axes
        rainbow_colors = cm.jet(np.linspace(0, ch_step, (ch_end - ch_start) + 1))
        x_ax.set_prop_cycle(cycler('color', rainbow_colors))
        x_ax_log.set_prop_cycle(cycler('color', rainbow_colors))
        z_ax.set_prop_cycle(cycler('color', rainbow_colors))
        z_ax_log.set_prop_cycle(cycler('color', rainbow_colors))

        x_data = file.data[file.data.Component == "X"]
        z_data = file.data[file.data.Component == "Z"]

        channels = [f'{num}' for num in range(1, len(file.ch_times) + 1)]
        min_ch = ch_start - 1
        max_ch = min(ch_end - 1, len(channels) - 1)
        plotting_channels = channels[min_ch: max_ch + 1: ch_step]
        if single_file is True:
            line_color = None

        for ind, ch in enumerate(plotting_channels):
            if incl_label is True:
                if single_file is True:
                    label = f"{file.ch_times[min_ch + ind]:.3f}ms"
                    # label = ch
                else:
                    if ind == 0:
                        label = name
                    else:
                        label = None
            else:
                label = None

            x = z_data.Station.astype(float) + station_shift
            zz = z_data.loc[:, ch].astype(float) * data_scaling  # * -1
            xx = x_data.loc[:, ch].astype(float) * data_scaling  # * -1

            if filter is True:
                zz = savgol_filter(zz, 21, 3)
                xx = savgol_filter(xx, 21, 3)

            for ax in [x_ax, x_ax_log]:
                ax.plot(x, xx,
                        color=line_color,
                        alpha=alpha,
                        # alpha=1 - (ind / (len(plotting_channels))) * 0.9,
                        label=label,
                        ls=ls,
                        zorder=1)
            for ax in [z_ax, z_ax_log]:
                ax.plot(x, zz,
                        color=line_color,
                        alpha=alpha,
                        # alpha=1 - (ind / (len(plotting_channels))) * 0.9,
                        label=label,
                        ls=ls,
                        zorder=1)

            for ax in axes:
                if x_min and x_max:
                    ax.set_xlim([x_min, x_max])
                else:
                    ax.set_xlim([x.min(), x.max()])
                if y_min and y_max:
                    ax.set_ylim([y_min, y_max])

    def format_figure(figure, title, files, min_ch, max_ch, ch_step=1, b_field=False, incl_footnote=False,
                      legend_times=None, incl_legend=True, incl_legend_ls=False, ylabel=''):
        for legend in figure.legends:
            legend.remove()

        x_ax, x_ax_log, z_ax, z_ax_log = figure.axes
        # Set the labels
        z_ax.set_xlabel(f"Station")
        z_ax_log.set_xlabel(f"Station")
        if ylabel:
            for ax in figure.axes:
                ax.set_ylabel(ylabel)
        else:
            if b_field is True:
                for ax in figure.axes:
                    ax.set_ylabel(f"EM Response\n(nT)")
            else:
                for ax in figure.axes:
                    ax.set_ylabel(f"EM Response\n(nT/s)")

        figure.suptitle(title)
        x_ax.set_title(f"X Component")
        z_ax.set_title(f"Z Component")
        x_ax_log.set_title(f"X Component")
        z_ax_log.set_title(f"Z Component")

        if incl_legend is True:
            # Create a manual legend
            if legend_times is not None:
                colors = cm.jet(np.linspace(0, 1, int(((max_ch - min_ch) + 1) / ch_step)))
                times = np.array(legend_times[min_ch - 1: max_ch: ch_step])
                handles = []
                labels = []
                for i, color in enumerate(colors):
                    line = Line2D([0], [0], color=color, linestyle="-")
                    label = f"{times[i]:.3f}ms"
                    handles.append(line)
                    labels.append(label)
            else:
                # Create a legend from the plotted lines
                handles, labels = z_ax.get_legend_handles_labels()

            # Add file linestyles to the legend for each different file type
            if incl_legend_ls:
                filetypes = []
                for file in files:
                    filetypes.append(get_filetype(file))
                lines = [Line2D([0], [0], color='k', linestyle=styles.get(filetype)) for filetype in filetypes]
                handles.extend(lines)
                labels.extend(filetypes)
                # figure.legend(manual_lines, manual_labels, loc='center right')

            figure.legend(handles, labels, loc='upper right')

        if incl_footnote is True:
            footnote = ''
            for file in files:
                footnote += f"{get_filetype(file)} file plotting channels {min_ch}-{max_ch}" \
                            f" ({file.ch_times[min_ch - 1]:.3f}ms-{file.ch_times[max_ch - 1]:.3f}ms).  "

            # Add the footnote
            z_ax.text(0.995, 0.01, footnote,
                      ha='right',
                      va='bottom',
                      size=6,
                      transform=figure.transFigure)

    def get_filetype(file_object):
        if isinstance(file_object, TEMFile):
            return "Maxwell"
        elif isinstance(file_object, MUNFile):
            return "MUN"
        elif isinstance(file_object, IRAPFile):
            return "IRAP"
        elif isinstance(file_object, PlateFFile):
            return "PLATE"
        else:
            raise TypeError(F"{file_object} is not a valid filetype.")

    def get_folder_range(folder, file_type, start_ch, end_ch):
        """Calculates the Max and Min Y values from all files in the folder"""
        print(F"Calculating maximum and minimum Y values in {folder} between channels {start_ch} and {end_ch}.")
        mins, maxes = [], []
        if file_type == "Maxwell":
            files = folder.glob("*.TEM")
            for file in files:
                tem_file = TEMFile().parse(file)
                mn, mx = tem_file.get_range(start_ch=start_ch, end_ch=end_ch)
                mins.append(mn)
                maxes.append(mx)
        elif file_type == "MUN":
            files = folder.glob("*.DAT")
            maxes, mins = [], []
            for file in files:
                tem_file = TEMFile().parse(file)
                mn, mx = tem_file.get_range(start_ch=start_ch, end_ch=end_ch)
                mins.append(mn)
                maxes.append(mx)
        print(F"Minimum Y: {min(mins):.2f}\nMaximum Y: {max(maxes):.2f}.")
        return min(mins), max(maxes)

    def get_residual_file(combined_file, folder, plotting_files):
        """
        Remove the sum of the data from the individual plate files that make up a combined model from
        the original file (combined_file)
        :param combined_file: file object
        :param folder: Path object, filder which contains the base files
        :param plotting_files: list, names of the files being plotted
        """

        def get_composite_base_files(file_obj, base_files):
            """Return the individual plate files that are in the target file"""
            plates = list(file_obj.filepath.stem)
            composite_files = []
            for base_file in base_files:
                if base_file in plates:
                    if isinstance(file_obj, TEMFile):
                        composite_files.append(folder.joinpath(base_file).with_suffix(".TEM"))
                    elif isinstance(file_obj, MUNFile):
                        composite_files.append(folder.joinpath(base_file).with_suffix(".DAT"))
                    else:
                        raise TypeError(F"{base_file} is an invalid file object.")
            print(f"Individual plate files in {file_obj.filepath.name}: {', '.join([b.name for b in composite_files])}.")
            return composite_files

        base_files = [f for f in plotting_files if len(f) == 1]
        channels = [f"CH{num}" for num in range(1, len(combined_file.ch_times) + 1)]
        residual_file = copy.deepcopy(combined_file)

        composite_files = get_composite_base_files(combined_file, base_files)
        print(f"Calculating the sum of the data from {', '.join([f.name for f in composite_files])}.")
        for file in composite_files:
            if file.suffix == ".TEM":
                file_obj = TEMFile().parse(file)
            elif file.suffix == ".DAT":
                file_obj = MUNFile().parse(file)
            else:
                raise TypeError(F"{file.suffix} is not yet supported.")

            residual_file.data.loc[:, channels] = residual_file.data.loc[:, channels] - \
                                                  file_obj.data.loc[:, channels]

        return residual_file

    def clear_axes(axes):
        for ax in axes:
            ax.clear()

    def log_scale(x_ax, z_ax):
        x_ax.set_yscale('symlog', subs=list(np.arange(2, 10, 1)), linthresh=10, linscale=1. / math.log(10))
        z_ax.set_yscale('symlog', subs=list(np.arange(2, 10, 1)), linthresh=10, linscale=1. / math.log(10))

    def get_runtime(t):
        return f"{math.floor((time.time() - t) / 60):02.0d}:{(time.time() - t) % 60:02.0d}"

    def plot_aspect_ratio():
        figure, ((x_ax, x_ax_log), (z_ax, z_ax_log)) = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='col')
        axes = [x_ax, z_ax, x_ax_log, z_ax_log]
        figure.set_size_inches((11 * 1.33, 8.5 * 1.33))

        maxwell_dir = sample_files.joinpath(r"Aspect Ratio\Maxwell\2m stations")
        mun_dir = sample_files.joinpath(r"Aspect Ratio\MUN")
        plate_dir = sample_files.joinpath(r"Aspect Ratio\PLATE\2m stations")
        irap_dir = sample_files.joinpath(r"Aspect Ratio\IRAP")

        global min_ch, max_ch, channel_step
        min_ch, max_ch = 21, 21
        channel_step = 1

        t = time.time()



        # tester = TestRunner()
        # tester.show()
        #
        # logging_file.write(f">>Plotting aspect ratio test results<<\n")
        #
        # # # Maxwell
        # # maxwell_dir = sample_files.joinpath(r"Aspect Ratio\Maxwell\2m stations")
        # # tester.add_row(str(maxwell_dir), "Maxwell")
        # # tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Data Scaling")).setText("0.000001")
        # # tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Station Shift")).setText("-400")
        # # tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Channel Start")).setText("21")
        # # tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Channel End")).setText("44")
        #
        # # MUN
        # mun_dir = sample_files.joinpath(r"Aspect Ratio\MUN")
        # tester.add_row(str(mun_dir), "MUN")
        # tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Station Shift")).setText("-200")
        # tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Channel Start")).setText("21")
        # tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Channel End")).setText("44")
        # tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Alpha")).setText("0.5")
        #
        # # # Plate
        # # plate_dir = sample_files.joinpath(r"Aspect Ratio\PLATE\2m stations")
        # # tester.add_row(str(plate_dir), "PLATE")
        # # tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Alpha")).setText("0.5")
        #
        # # Peter
        # irap_dir = sample_files.joinpath(r"Aspect Ratio\IRAP")
        # tester.add_row(str(irap_dir), "IRAP")
        # tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Channel Start")).setText("21")
        # tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Alpha")).setText("0.5")
        #
        # """ Plotting """
        # tester.plot_profiles_rbtn.setChecked(True)
        #
        # tester.custom_stations_cbox.setChecked(False)
        # tester.y_cbox.setChecked(True)
        # tester.test_name_edit.setText(r"Aspect Ratio")
        # # tester.fixed_range_cbox.setChecked(True)
        # tester.include_edit.setText("150")
        # tester.include_edit.editingFinished.emit()
        #
        # # file.write(f"Plotting 150m plates (linear, all stations)\n")
        # # pdf_file = str(sample_files.joinpath(r"Aspect Ratio\Aspect Ratio - 150m plate.PDF"))
        # # tester.output_filepath_edit.setText(pdf_file)
        # # tester.print_pdf(from_script=True)
        #
        # logging_file.write(f"Plotting 150m plates (linear, stations 0-200)\n")
        # pdf_file = str(sample_files.joinpath(r"Aspect Ratio\Aspect Ratio - 150m plate (Station 0-200).PDF"))
        # tester.custom_stations_cbox.setChecked(True)
        # tester.station_start_sbox.setValue(0)
        # tester.station_end_sbox.setValue(200)
        # tester.output_filepath_edit.setText(pdf_file)
        # tester.print_pdf(from_script=True)
        #
        # # file.write(f"Plotting 600m plates (linear, all stations)\n")
        # # pdf_file = str(sample_files.joinpath(r"Aspect Ratio\Aspect Ratio - 600m plate.PDF"))
        # # tester.custom_stations_cbox.setChecked(False)
        # # tester.output_filepath_edit.setText(pdf_file)
        # # tester.include_edit.setText("600")
        # # tester.include_edit.editingFinished.emit()
        # # tester.print_pdf(from_script=True)
        #
        # logging_file.write(f"Plotting 600m plates (linear, stations 0-200)\n")
        # pdf_file = str(sample_files.joinpath(r"Aspect Ratio\Aspect Ratio - 600m plate (Station 0-200).PDF"))
        # tester.custom_stations_cbox.setChecked(True)
        # tester.output_filepath_edit.setText(pdf_file)
        # tester.print_pdf(from_script=True)
        #
        # """ Log Y """
        # tester.log_y_cbox.setChecked(True)
        # tester.custom_stations_cbox.setChecked(False)
        #
        # # file.write(f"Plotting 150m plates (log, all stations)\n")
        # # pdf_file = str(sample_files.joinpath(r"Aspect Ratio\Aspect Ratio - 150m plate [LOG].PDF"))
        # # tester.output_filepath_edit.setText(pdf_file)
        # # tester.print_pdf(from_script=True)
        #
        # logging_file.write(f"Plotting 150m plates (log, stations 0-200)\n")
        # pdf_file = str(sample_files.joinpath(r"Aspect Ratio\Aspect Ratio - 150m plate (Station 0-200) [LOG].PDF"))
        # tester.custom_stations_cbox.setChecked(True)
        # tester.station_start_sbox.setValue(0)
        # tester.station_end_sbox.setValue(200)
        # tester.output_filepath_edit.setText(pdf_file)
        # tester.print_pdf(from_script=True)
        #
        # # file.write(f"Plotting 600m plates (log, all stations)\n")
        # # pdf_file = str(sample_files.joinpath(r"Aspect Ratio\Aspect Ratio - 600m plate [LOG].PDF"))
        # # tester.custom_stations_cbox.setChecked(False)
        # # tester.output_filepath_edit.setText(pdf_file)
        # # tester.include_edit.setText("600")
        # # tester.include_edit.editingFinished.emit()
        # # tester.print_pdf(from_script=True)
        #
        # logging_file.write(f"Plotting 600m plates (log, stations 0-200)\n")
        # pdf_file = str(sample_files.joinpath(r"Aspect Ratio\Aspect Ratio - 600m plate (Station 0-200) [LOG].PDF"))
        # tester.custom_stations_cbox.setChecked(True)
        # tester.output_filepath_edit.setText(pdf_file)
        # tester.print_pdf(from_script=True)

        print(f"Aspect ratio plot time: {get_runtime(t)}")
        logging_file.write(f"Aspect ratio plot time: {get_runtime(t)}\n")

    def plot_two_way_induction():
        t = time.time()
        tester = TestRunner()
        tester.show()

        tester.custom_stations_cbox.setChecked(False)
        tester.plot_profiles_rbtn.setChecked(True)
        tester.y_cbox.setChecked(False)
        tester.test_name_edit.setText(r"Two-Way Induction")

        # Maxwell
        maxwell_dir = sample_files.joinpath(r"Two-way induction\300x100\100S\Maxwell")
        tester.add_row(str(maxwell_dir), "Maxwell")
        tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Data Scaling")).setText("0.000001")
        # tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Station Shift")).setText("-400")
        tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Channel Start")).setText("21")
        tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Channel End")).setText("44")

        # MUN
        mun_dir = sample_files.joinpath(r"Two-way induction\300x100\100S\MUN")
        tester.add_row(str(mun_dir), "MUN")
        tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Station Shift")).setText("300")
        tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Channel Start")).setText("21")
        tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Channel End")).setText("44")
        tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Alpha")).setText("0.5")

        # Plate
        plate_dir = sample_files.joinpath(r"Two-way induction\300x100\100S\PLATE")
        tester.add_row(str(plate_dir), "PLATE")
        tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Alpha")).setText("0.5")

        # # Peter
        # irap_dir = sample_files.joinpath(r"Two-way induction\300x100\100S\IRAP")
        # tester.add_row(str(irap_dir), "IRAP")
        # tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Channel Start")).setText("21")
        # tester.table.item(tester.table.rowCount() - 1, tester.header_labels.index("Alpha")).setText("0.5")

        """Plotting"""
        pdf_file = str(sample_files.joinpath(r"Two-way induction\300x100\Two-Way Induction - 100S.PDF"))
        tester.output_filepath_edit.setText(pdf_file)
        tester.print_pdf()

        pdf_file = str(sample_files.joinpath(r"Two-way induction\300x100\Two-Way Induction - 100S (Station 0-200).PDF"))
        tester.custom_stations_cbox.setChecked(True)
        tester.station_start_sbox.setValue(0)
        tester.station_end_sbox.setValue(200)
        tester.output_filepath_edit.setText(pdf_file)
        tester.print_pdf()

        """LOG Y"""
        tester.log_y_cbox.setChecked(True)
        tester.custom_stations_cbox.setChecked(False)
        pdf_file = str(sample_files.joinpath(r"Two-way induction\300x100\Two-Way Induction - 100S [LOG].PDF"))
        tester.output_filepath_edit.setText(pdf_file)
        tester.print_pdf()

        pdf_file = str(sample_files.joinpath(r"Two-way induction\300x100\Two-Way Induction - 100S (Station 0-200) [LOG].PDF"))
        tester.custom_stations_cbox.setChecked(True)
        tester.station_start_sbox.setValue(0)
        tester.station_end_sbox.setValue(200)
        tester.output_filepath_edit.setText(pdf_file)
        tester.print_pdf()

        print(f"Total time: {math.floor((time.time() - t) / 60):02.0f}:{(time.time() - t) % 60:.0f}")

    def plot_run_on_comparison():
        tester = TestRunner()
        tester.show()

        """Run the run-on effects tests"""
        tester.plot_run_on_comparison_rbtn.setChecked(True)

        tester.test_name_edit.setText("Maxwell Run-on Effect Calculation")
        tester.add_row(folderpath=str(sample_files.joinpath(r"Run-on effect\600x600C")),
                       file_type='Maxwell')

        tester.table.item(0, 2).setText("0.000001")
        # tester.table.item(0, 4).setText("6")
        # tester.table.item(0, 5).setText("44")
        tester.table.item(0, 4).setText("45")
        tester.table.item(0, 5).setText("68")

        # tester.include_edit.setText("150, B")
        # tester.include_edit.editingFinished.emit()
        # tester.output_filepath_edit.setText(
        #     str(sample_files.joinpath(r"Run-on effect\Run on effect - 150m plate, 1,000 S.PDF")))
        # tester.print_pdf()
        #
        # tester.include_edit.setText("150, C")
        # tester.include_edit.editingFinished.emit()
        # tester.output_filepath_edit.setText(
        #     str(sample_files.joinpath(r"Run-on effect\Run on effect - 150m plate, 10,000 S.PDF")))
        # tester.print_pdf()
        #
        # tester.include_edit.setText("600, B")
        # tester.include_edit.editingFinished.emit()
        # tester.output_filepath_edit.setText(
        #     str(sample_files.joinpath(r"Run-on effect\Run on effect - 600m plate, 1,000 S.PDF")))
        # tester.print_pdf()
        #
        tester.include_edit.setText("600, C")
        tester.include_edit.editingFinished.emit()
        tester.output_filepath_edit.setText(
            str(sample_files.joinpath(r"Run-on effect\On-time formula.PDF")))
            # str(sample_files.joinpath(r"Run-on effect test\Run on effect - 600m plate, 10,000 S, full waveform.PDF")))
        tester.print_pdf()

    def plot_run_on_convergence():
        tester = TestRunner()
        tester.show()
        """Plot the half-cycle convergence of run-on effect"""
        tester.plot_run_on_convergence_rbtn.setChecked(True)

        tester.test_name_edit.setText("Run-on Effect Convergence")
        tester.add_row(folderpath=str(sample_files.joinpath(r"Run-on effect\100s")),
                       file_type='Maxwell')
        tester.table.item(0, 2).setText("0.000001")

        tester.output_filepath_edit.setText(str(sample_files.joinpath(
            r"Run-on effect\Run-on convergence - 150m plate, 1,000 S.PDF")))
        tester.include_edit.setText("150, B")
        tester.include_edit.editingFinished.emit()
        tester.print_pdf()
        #
        # tester.output_filepath_edit.setText(str(sample_files.joinpath(
        #     r"Run-on effect\Run-on convergence - 150m plate, 10,000 S.PDF")))
        # tester.include_edit.setText("150, C")
        # tester.include_edit.editingFinished.emit()
        # tester.print_pdf()

        tester.output_filepath_edit.setText(str(sample_files.joinpath(
            r"Run-on effect\Run-on convergence - 600m plate, 1,000 S.PDF")))
        tester.include_edit.setText("600, B")
        tester.include_edit.editingFinished.emit()
        tester.print_pdf()

        # tester.output_filepath_edit.setText(str(sample_files.joinpath(
        #     r"Run-on effect\Run-on convergence - 600m plate, 10,000 S.PDF")))
        # tester.include_edit.setText("600, C")
        # tester.include_edit.editingFinished.emit()
        # tester.print_pdf()

    def tabulate_run_on_convergence():
        tester = TestRunner()
        tester.show()
        """Tabulate the number of half-cycles required for convergence of run-on effect"""
        tester.table_run_on_convergence_rbtn.setChecked(True)

        tester.test_name_edit.setText("Run-on Effect Convergence")
        tester.add_row(folderpath=str(sample_files.joinpath(r"Run-on effect\100s")),
                       file_type='Maxwell')
        tester.table.item(0, 2).setText("0.000001")

        # tester.output_filepath_edit.setText(str(sample_files.joinpath(
        #     r"Run-on effect\Run-on Effect Convergence - 150m plate, 1,000 S.CSV")))
        # tester.include_edit.setText("150, B")
        # tester.include_edit.editingFinished.emit()
        # tester.print_pdf()

        tester.output_filepath_edit.setText(str(sample_files.joinpath(
            r"Run-on effect\Run-on Effect Convergence - 150m plate, 10,000 S.CSV")))
        tester.include_edit.setText("150, C")
        tester.include_edit.editingFinished.emit()
        tester.print_pdf()

        # tester.output_filepath_edit.setText(str(sample_files.joinpath(
        #     r"Run-on effect\Run-on Effect Convergence - 600m plate, 1,000 S.CSV")))
        # tester.include_edit.setText("600, B")
        # tester.include_edit.editingFinished.emit()
        # tester.print_pdf()

        tester.output_filepath_edit.setText(str(sample_files.joinpath(
            r"Run-on effect\Run-on Effect Convergence - 600m plate, 10,000 S.CSV")))
        tester.include_edit.setText("600, C")
        tester.include_edit.editingFinished.emit()
        tester.print_pdf()

    def compare_maxwell_ribbons():
        output = sample_files.joinpath(r"Infinite Thin Sheet\Infinite Thin Sheet Ribbon Comparison.PDF")
        figure, (x_ax, z_ax) = plt.subplots(nrows=2, ncols=1, sharex='all', sharey="all")
        figure.set_size_inches((8.5, 11))
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.2)

        def plot(filepath, color, ch_start, ch_end, name="", station_shift=0, data_scaling=1., alpha=1.,
                 incl_footnote=False):

            x_ax.set_yscale('symlog', subs=list(np.arange(2, 10, 1)), linthresh=10, linscale=1. / math.log(10))
            z_ax.set_yscale('symlog', subs=list(np.arange(2, 10, 1)), linthresh=10, linscale=1. / math.log(10))

            parser = TEMFile()
            file = parser.parse(filepath)

            print(f"Plotting {filepath.name}.")

            x_data = file.data[file.data.COMPONENT == "X"]
            y_data = file.data[file.data.COMPONENT == "Y"]
            z_data = file.data[file.data.COMPONENT == "Z"]

            channels = [f'CH{num}' for num in range(1, len(file.ch_times) + 1)]
            min_ch = ch_start - 1
            max_ch = min(ch_end - 1, len(channels) - 1)
            plotting_channels = channels[min_ch: max_ch + 1]
            global footnote

            for ind, ch in enumerate(plotting_channels):
                if ind == 0:
                    label = f"{name}"

                    if incl_footnote:
                        if min_ch == max_ch:
                            footnote += f"Maxwell file plotting channel {min_ch + 1} ({file.ch_times[max_ch]:.3f}ms).  "
                        else:
                            footnote += f"Maxwell file plotting channels {min_ch + 1}-{max_ch + 1}" \
                                             f" ({file.ch_times[min_ch]:.3f}ms-{file.ch_times[max_ch]:.3f}ms).  "
                else:
                    label = None

                x = x_data.STATION.astype(float) + station_shift
                xx = x_data.loc[:, ch].astype(float) * data_scaling
                yy = y_data.loc[:, ch].astype(float) * data_scaling
                zz = z_data.loc[:, ch].astype(float) * data_scaling

                x_ax.plot(x, xx,
                          color=color,
                          alpha=alpha,
                          label=label,
                          zorder=1)
                z_ax.plot(x, zz,
                          color=color,
                          alpha=alpha,
                          label=label,
                          zorder=1)

        folder_10 = Path(sample_files.joinpath(r"Infinite Thin Sheet\Maxwell\10 Ribbons"))
        folder_50 = Path(sample_files.joinpath(r"Infinite Thin Sheet\Maxwell\50 Ribbons"))

        files_10 = os_sorted(list(folder_10.glob("*.tem")))
        files_50 = os_sorted(list(folder_50.glob("*.tem")))

        count = 0
        with PdfPages(output) as pdf:
            for filepath_10, filepath_50 in zip(files_10, files_50):
                print(f"Plotting set {count + 1}/{len(files_10)}")
                global footnote
                footnote = ''

                # Plot the files
                plot(filepath_10, "b", 21, 44, name="10 Ribbons", station_shift=0, data_scaling=.000001, alpha=1.,
                     incl_footnote=True)
                plot(filepath_50, "r", 21, 44, name="50 Ribbons", station_shift=0, data_scaling=.000001, alpha=0.5,
                     incl_footnote=False)

                # Set the labels
                z_ax.set_xlabel(f"Station")
                x_ax.set_ylabel(f"EM Response\n(nT/s)")
                z_ax.set_ylabel(f"EM Response\n(nT/s)")
                plt.suptitle(f"Infinite Thin Sheet Ribbon Comparison")
                x_ax.set_title(f"{filepath_10.stem} (X Component)")
                z_ax.set_title(f"{filepath_10.stem} (Z Component)")

                # Create the legend
                handles, labels = x_ax.get_legend_handles_labels()

                # sort both labels and handles by labels
                # labels, handles = zip(*os_sorted(zip(labels, handles), key=lambda t: t[0]))
                figure.legend(handles, labels)

                # Add the footnote
                z_ax.text(0.995, 0.01, footnote,
                          ha='right',
                          va='bottom',
                          size=6,
                          transform=figure.transFigure)

                # plt.show()
                pdf.savefig(figure, orientation='portrait')
                x_ax.clear()
                z_ax.clear()

                count += 1

        print(f"Process complete.")
        os.startfile(output)

    def compare_step_on_b_with_theory():
        figure, (x_ax, z_ax) = plt.subplots(nrows=2, sharex='all')
        figure.set_size_inches((8.5, 11))

        def plot_maxwell(filepath, color, ch_start, ch_end, name="", station_shift=0, data_scaling=1., alpha=1.):

            parser = TEMFile()
            file = parser.parse(filepath)

            print(f"Plotting {filepath.name}.")

            x_data = file.data[file.data.COMPONENT == "X"]
            z_data = file.data[file.data.COMPONENT == "Z"]

            channels = [f'CH{num}' for num in range(1, len(file.ch_times) + 1)]
            min_ch = ch_start - 1
            max_ch = min(ch_end - 1, len(channels) - 1)
            plotting_channels = channels[min_ch: max_ch + 1]

            for ind, ch in enumerate(plotting_channels):
                if ind == 0:
                    label = f"{name}"
                    global footnote

                    if min_ch == max_ch:
                        footnote += f"Maxwell file plotting channel {min_ch + 1} ({file.ch_times[max_ch]:.3f}ms).  "
                    else:
                        footnote += f"Maxwell file plotting channels {min_ch + 1}-{max_ch + 1}" \
                                    f" ({file.ch_times[min_ch]:.3f}ms-{file.ch_times[max_ch]:.3f}ms).  "
                else:
                    label = None

                x = z_data.STATION.astype(float) + station_shift
                zz = z_data.loc[:, ch].astype(float) * data_scaling  # * -1
                xx = x_data.loc[:, ch].astype(float) * data_scaling  # * -1

                x_ax.plot(x, xx,
                          color=color,
                          linestyle="-",
                          # alpha=alpha,
                          alpha=1 - (ind / (len(plotting_channels))) * 0.9,
                          label=label,
                          zorder=1)
                z_ax.plot(x, zz,
                          color=color,
                          linestyle="-",
                          # alpha=alpha,
                          alpha=1 - (ind / (len(plotting_channels))) * 0.9,
                          label=label,
                          zorder=1)

                x_ax.set_xlim([x.min(), x.max()])
                z_ax.set_xlim([x.min(), x.max()])

        def plot_theory(theory_x_file, theory_z_file):
            x_df = pd.read_excel(theory_x_file, header=4).dropna(axis=1)
            z_df = pd.read_excel(theory_z_file, header=4).dropna(axis=1)
            x = x_df.Position
            global footnote
            footnote += f"Theory plotting {(x_df.columns[1] * 1e3):.3f}ms to {(x_df.columns[-1] * 1e3):.3f}ms"

            for ind, (_, ch_response) in enumerate(x_df.iloc[:, 1:].iteritems()):
                if ind == 0:
                    label = f"Theory"
                else:
                    label = None

                theory_x = ch_response.values

                x_ax.plot(x, theory_x,
                          color="r",
                          linestyle="-",
                          alpha=1 - (ind / (len(x_df.columns) - 1)) * 0.9,
                          label=label,
                          zorder=1)

            for ind, (_, ch_response) in enumerate(z_df.iloc[:, 1:].iteritems()):
                if ind == 0:
                    label = f"Theory"
                else:
                    label = None

                theory_z = ch_response.values

                z_ax.plot(x, theory_z,
                          color="r",
                          linestyle="-",
                          alpha=1 - (ind / (len(x_df.columns) - 1)) * 0.9,
                          label=label,
                          zorder=1)

        def format_figure(title, footnote, b_field=False):
            # for text in figure.texts:
            #     text.remove()
            #
            for legend in figure.legends:
                legend.remove()

            # Set the labels
            z_ax.set_xlabel(f"Station")
            if b_field is True:
                x_ax.set_ylabel(f"EM Response\n(nT)")
                z_ax.set_ylabel(f"EM Response\n(nT)")
            else:
                x_ax.set_ylabel(f"EM Response\n(nT/s)")
                z_ax.set_ylabel(f"EM Response\n(nT/s)")
            figure.suptitle(title)
            x_ax.set_title(f"X Component")
            z_ax.set_title(f"Z Component")

            # Create the legend
            handles, labels = z_ax.get_legend_handles_labels()

            # sort both labels and handles by labels
            figure.legend(handles, labels)

            # Add the footnote
            z_ax.text(0.995, 0.01, footnote,
                      ha='right',
                      va='bottom',
                      size=6,
                      transform=figure.transFigure)

        def plot(theory_x_file, theory_z_file, maxwell_folder, conductance, b_field=False, log=False):
            assert theory_x_file.is_file(), F"Theory file X does not exist."
            assert theory_z_file.is_file(), F"Theory file Z does not exist."
            assert maxwell_folder.is_dir(), F"Maxwell folder does not exist."
            files = os_sorted(list(maxwell_folder.glob(f"*{conductance}.tem")))

            count = 0
            for filepath in files:
                print(f"Plotting set {count + 1}/{len(files)}")
                if log:
                    x_ax.set_yscale('symlog', subs=list(np.arange(2, 10, 1)), linthresh=10, linscale=1. / math.log(10))
                    z_ax.set_yscale('symlog', subs=list(np.arange(2, 10, 1)), linthresh=10, linscale=1. / math.log(10))

                global footnote
                footnote = ''

                # Plot the files
                plot_maxwell(filepath, "b", 1, 100, name=f"{filepath.name}", station_shift=0, data_scaling=1., alpha=1.)
                plot_theory(theory_x_file, theory_z_file)
                if b_field is True:
                    format_figure(f"Infinite Thin Sheet B-field Current Step-On - {conductance}S", footnote,
                                  b_field=True)
                else:
                    format_figure(f"Infinite Thin Sheet dB/dt Current Step-On - {conductance}S", footnote,
                                  b_field=False)

                pdf.savefig(figure, orientation='portrait')

                x_ax.clear()
                z_ax.clear()

                count += 1

        # """ B FIELD """
        # """1 S"""
        # output = sample_files.joinpath(r"Infinite Thin Sheet\Infinite Thin Sheet B-field Step-on Comparison - 1S.PDF")
        # with PdfPages(output) as pdf:
        #     theory_x_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\B\Infinite sheet 1S B X.xlsx")
        #     theory_z_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\B\Infinite sheet 1S B Z.xlsx")
        #     maxwell_folder = sample_files.joinpath(r"Infinite Thin Sheet\Maxwell\B")
        #
        #     plot(theory_x_file, theory_z_file, maxwell_folder, "1", b_field=True, log=False)
        #     os.startfile(output)
        #
        # """10 S"""
        # output = sample_files.joinpath(r"Infinite Thin Sheet\Infinite Thin Sheet B-field Step-on Comparison - 10S.PDF")
        # with PdfPages(output) as pdf:
        #     theory_x_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\B\Infinite sheet 10S B X.xlsx")
        #     theory_z_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\B\Infinite sheet 10S B Z.xlsx")
        #     maxwell_folder = sample_files.joinpath(r"Infinite Thin Sheet\Maxwell\B")
        #
        #     plot(theory_x_file, theory_z_file, maxwell_folder, "10", b_field=True, log=False)
        #     os.startfile(output)
        #
        # """100 S"""
        # output = sample_files.joinpath(r"Infinite Thin Sheet\Infinite Thin Sheet B-field Step-on Comparison - 100S.PDF")
        # with PdfPages(output) as pdf:
        #     theory_x_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\B\Infinite sheet 100S B X.xlsx")
        #     theory_z_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\B\Infinite sheet 100S B Z.xlsx")
        #     maxwell_folder = sample_files.joinpath(r"Infinite Thin Sheet\Maxwell\B")
        #
        #     plot(theory_x_file, theory_z_file, maxwell_folder, "100", b_field=True, log=False)
        #     os.startfile(output)
        #
        # """ Log Scale """
        # """1 S"""
        # output = sample_files.joinpath(
        #     r"Infinite Thin Sheet\Infinite Thin Sheet B-field Step-on Comparison - 1S (log).PDF")
        # with PdfPages(output) as pdf:
        #     theory_x_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\B\Infinite sheet 1S B X.xlsx")
        #     theory_z_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\B\Infinite sheet 1S B Z.xlsx")
        #     maxwell_folder = sample_files.joinpath(r"Infinite Thin Sheet\Maxwell\B")
        #
        #     plot(theory_x_file, theory_z_file, maxwell_folder, "1", b_field=True, log=True)
        #     os.startfile(output)
        #
        # """10 S"""
        # output = sample_files.joinpath(
        #     r"Infinite Thin Sheet\Infinite Thin Sheet B-field Step-on Comparison - 10S (log).PDF")
        # with PdfPages(output) as pdf:
        #     theory_x_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\B\Infinite sheet 10S B X.xlsx")
        #     theory_z_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\B\Infinite sheet 10S B Z.xlsx")
        #     maxwell_folder = sample_files.joinpath(r"Infinite Thin Sheet\Maxwell\B")
        #
        #     plot(theory_x_file, theory_z_file, maxwell_folder, "10", b_field=True, log=True)
        #     os.startfile(output)
        #
        # """100 S"""
        # output = sample_files.joinpath(
        #     r"Infinite Thin Sheet\Infinite Thin Sheet B-field Step-on Comparison - 100S (log).PDF")
        # with PdfPages(output) as pdf:
        #     theory_x_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\B\Infinite sheet 100S B X.xlsx")
        #     theory_z_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\B\Infinite sheet 100S B Z.xlsx")
        #     maxwell_folder = sample_files.joinpath(r"Infinite Thin Sheet\Maxwell\B")
        #
        #     plot(theory_x_file, theory_z_file, maxwell_folder, "100", b_field=True, log=True)
        #     os.startfile(output)

        """ dBdt """
        """1 S"""
        output = sample_files.joinpath(r"Infinite Thin Sheet\Infinite Thin Sheet dBdt Step-on Comparison - 1S.PDF")
        with PdfPages(output) as pdf:
            theory_x_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\dBdt\Infinite sheet 1S dBdt X.xlsx")
            theory_z_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\dBdt\Infinite sheet 1S dBdt Z.xlsx")
            maxwell_folder = sample_files.joinpath(r"Infinite Thin Sheet\Maxwell\dBdt")

            plot(theory_x_file, theory_z_file, maxwell_folder, "1", b_field=False, log=False)
            os.startfile(output)

        # """10 S"""
        # output = sample_files.joinpath(r"Infinite Thin Sheet\Infinite Thin Sheet dBdt Step-on Comparison - 10S.PDF")
        # with PdfPages(output) as pdf:
        #     theory_x_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\dBdt\Infinite sheet 10S dBdt X.xlsx")
        #     theory_z_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\dBdt\Infinite sheet 10S dBdt Z.xlsx")
        #     maxwell_folder = sample_files.joinpath(r"Infinite Thin Sheet\Maxwell\dBdt")
        #
        #     plot(theory_x_file, theory_z_file, maxwell_folder, "10", b_field=False, log=False)
        #     os.startfile(output)
        #
        # """100 S"""
        # output = sample_files.joinpath(r"Infinite Thin Sheet\Infinite Thin Sheet dBdt Step-on Comparison - 100S.PDF")
        # with PdfPages(output) as pdf:
        #     theory_x_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\dBdt\Infinite sheet 100S dBdt X.xlsx")
        #     theory_z_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\dBdt\Infinite sheet 100S dBdt Z.xlsx")
        #     maxwell_folder = sample_files.joinpath(r"Infinite Thin Sheet\Maxwell\dBdt")
        #
        #     plot(theory_x_file, theory_z_file, maxwell_folder, "100", b_field=False, log=False)
        #     os.startfile(output)
        #
        # """ Log Scale """
        # """1 S"""
        # output = sample_files.joinpath(
        #     r"Infinite Thin Sheet\Infinite Thin Sheet dBdt Step-on Comparison - 1S (log).PDF")
        # with PdfPages(output) as pdf:
        #     theory_x_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\dBdt\Infinite sheet 1S dBdt X.xlsx")
        #     theory_z_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\dBdt\Infinite sheet 1S dBdt Z.xlsx")
        #     maxwell_folder = sample_files.joinpath(r"Infinite Thin Sheet\Maxwell\dBdt")
        #
        #     plot(theory_x_file, theory_z_file, maxwell_folder, "1", b_field=False, log=True)
        #     os.startfile(output)
        #
        # """10 S"""
        # output = sample_files.joinpath(
        #     r"Infinite Thin Sheet\Infinite Thin Sheet dBdt Step-on Comparison - 10S (log).PDF")
        # with PdfPages(output) as pdf:
        #     theory_x_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\dBdt\Infinite sheet 10S dBdt X.xlsx")
        #     theory_z_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\dBdt\Infinite sheet 10S dBdt Z.xlsx")
        #     maxwell_folder = sample_files.joinpath(r"Infinite Thin Sheet\Maxwell\dBdt")
        #
        #     plot(theory_x_file, theory_z_file, maxwell_folder, "10", b_field=False, log=True)
        #     os.startfile(output)
        #
        # """100 S"""
        # output = sample_files.joinpath(
        #     r"Infinite Thin Sheet\Infinite Thin Sheet dBdt Step-on Comparison - 100S (log).PDF")
        # with PdfPages(output) as pdf:
        #     theory_x_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\dBdt\Infinite sheet 100S dBdt X.xlsx")
        #     theory_z_file = sample_files.joinpath(r"Infinite Thin Sheet\Theory\dBdt\Infinite sheet 100S dBdt Z.xlsx")
        #     maxwell_folder = sample_files.joinpath(r"Infinite Thin Sheet\Maxwell\dBdt")
        #
        #     plot(theory_x_file, theory_z_file, maxwell_folder, "100", b_field=False, log=True)
        #     os.startfile(output)

    def plot_df(axes, data, reference_file, ch_step=1, ls='-'):
        x_ax, z_ax, x_ax_log, z_ax_log = axes
        rainbow_colors = cm.jet(np.linspace(0, ch_step, (max_ch - min_ch) + 1))
        x_ax.set_prop_cycle(cycler('color', rainbow_colors))
        x_ax_log.set_prop_cycle(cycler('color', rainbow_colors))
        z_ax.set_prop_cycle(cycler('color', rainbow_colors))
        z_ax_log.set_prop_cycle(cycler('color', rainbow_colors))

        x_data = data[data.COMPONENT == "X"]
        z_data = data[data.COMPONENT == "Z"]

        for ind, ch in enumerate(channels):
            label = f"{reference_file.ch_times[min_ch + (ind * ch_step)]:.3f}ms"

            x = z_data.STATION.astype(float)
            zz = z_data.loc[:, ch].astype(float)
            xx = x_data.loc[:, ch].astype(float)

            for ax in [x_ax, x_ax_log]:
                ax.plot(x, xx,
                        label=label,
                        ls=ls,
                        zorder=1)
            for ax in [z_ax, z_ax_log]:
                ax.plot(x, zz,
                        label=label,
                        ls=ls,
                        zorder=1)

    def plot_overburden():

        def calc_residual(combined_file, ob_file, plate_file):
            # Works for both MUN and Maxwell
            print(f"Calculating residual for {', '.join([f.filepath.name for f in [combined_file, ob_file, plate_file]])}")
            residual_file = copy.deepcopy(combined_file)
            channels = [f'CH{num}' for num in range(1, len(ob_file.ch_times) + 1)]

            calculated_data = ob_file.data.loc[:, channels] + plate_file.data.loc[:, channels]
            residual_data = combined_file.data.loc[:, channels] - calculated_data
            residual_file.data.loc[:, channels] = residual_data
            return residual_file

        def plot_overburden_and_plates(ch_step=1):
            """ Plot the overburden on its own """

            def plot_plates(ch_step=1):
                print(f">> Plotting plates alone")
                # Plot the plate models on their own
                for maxwell_file, mun_file, title in zip([maxwell_plate1_file, maxwell_plate2_file],
                                                         [mun_plate1_file, mun_plate2_file],
                                                         ["Plate 1 Only", "Plate 2 Only"]):
                    plot_max(axes,
                             maxwell_file,
                             min_ch,
                             max_ch,
                             ch_step=ch_step,
                             name="Maxwell",
                             alpha=0.6,
                             line_color=None,
                             ls=styles.get("Maxwell"),
                             data_scaling=1e-6,
                             single_file=True,
                             incl_label=True)

                    plot_mun(axes,
                             mun_file,
                             min_ch,
                             max_ch,
                             ch_step=ch_step,
                             name="MUN",
                             alpha=1.,
                             line_color=None,
                             ls=styles.get("MUN"),
                             single_file=True,
                             incl_label=False)

                    format_figure(figure,
                                  "Overburden Models\n" + title, [maxwell_file, mun_file],
                                  min_ch,
                                  max_ch,
                                  incl_legend=True,
                                  extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
                                  extra_labels=["Maxwell", "MUN"])

                    pdf.savefig(figure, orientation='landscape')
                    clear_axes(axes)
                    log_scale(x_ax_log, z_ax_log)

            def plot_overburden(ch_step=1):
                print(f">> Plotting overburden alone ({conductance})")
                plot_max(axes,
                         maxwell_ob_file,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name="Maxwell",
                         alpha=0.6,
                         line_color=None,
                         ls=styles.get("Maxwell"),
                         data_scaling=1e-6,
                         single_file=True,
                         incl_label=True)

                plot_mun(axes,
                         mun_ob_file,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name="MUN",
                         alpha=1.,
                         line_color=None,
                         ls=styles.get("MUN"),
                         single_file=True,
                         incl_label=False)

                format_figure(figure,
                              f"Overburden Models\n{conductance} Overburden Only", [maxwell_ob_file, mun_ob_file],
                              min_ch,
                              max_ch,
                              incl_legend=True,
                              extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
                              extra_labels=["Maxwell", "MUN"])

                pdf.savefig(figure, orientation='landscape')
                clear_axes(axes)
                log_scale(x_ax_log, z_ax_log)

            out_pdf = maxwell_folder.parents[1].joinpath(r"Overburden Model - Plates & Overburden Only.PDF")
            with PdfPages(out_pdf) as pdf:
                plot_plates(ch_step=ch_step)

                for conductance in ["1S", "10S"]:
                    maxwell_ob_file = TEMFile().parse(Path(maxwell_folder).joinpath(fr"{conductance} Overburden Only - 50m.TEM"))
                    mun_ob_file = MUNFile().parse(Path(mun_folder).joinpath(fr"overburden_{conductance}_V1000m_dBdt.DAT"))
                    plot_overburden(ch_step=ch_step)

            os.startfile(out_pdf)

        def plot_contact_effect(ch_step=1):
            """Effects of plate contact"""

            def plot_maxwell_contact_effect():
                print(F">>Plotting Maxwell contact effect ({conductance})")
                # Plot the in-contact plate with separated plate for each method
                plot_max(axes,
                         maxwell_comb_sep_file1,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name="Separated",
                         line_color=None,
                         ls='--',
                         data_scaling=1e-6,
                         alpha=1.,
                         single_file=True,
                         incl_label=False)
                plot_max(axes,
                         maxwell_comb_con_file1,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name="Contact",
                         line_color=None,
                         ls='-',
                         data_scaling=1e-6,
                         single_file=True,
                         incl_label=True)
                format_figure(figure,
                              f"Overburden Models\n"
                              f"Maxwell Plate Contact vs Separation [{conductance} Overburden with Plate 1]",
                              [maxwell_comb_sep_file1],
                              min_ch,
                              max_ch,
                              incl_legend=True,
                              extra_handles=["--", "-"],
                              extra_labels=["Separated", "Contact"])
                pdf.savefig(figure, orientation='landscape')
                clear_axes(axes)
                log_scale(x_ax_log, z_ax_log)

                # Plot the in-contact plate with separated plate for each method
                plot_max(axes,
                         maxwell_comb_sep_file2,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name="Separated",
                         line_color=None,
                         ls='--',
                         data_scaling=1e-6,
                         alpha=1.,
                         single_file=True,
                         incl_label=False)
                plot_max(axes,
                         maxwell_comb_con_file2,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name="Contact",
                         line_color=None,
                         ls='-',
                         data_scaling=1e-6,
                         single_file=True,
                         incl_label=True)
                format_figure(figure,
                              f"Overburden Models\n"
                              f"Maxwell Plate Contact vs Separation [{conductance} Overburden with Plate 2]",
                              [maxwell_comb_sep_file2],
                              min_ch,
                              max_ch,
                              incl_legend=True,
                              extra_handles=["--", "-"],
                              extra_labels=["Separated", "Contact"])
                pdf.savefig(figure, orientation='landscape')
                clear_axes(axes)
                log_scale(x_ax_log, z_ax_log)

            def plot_mun_contact_effect():
                print(F">>Plotting MUN contact effect ({conductance})")
                plot_mun(axes,
                         mun_comb_sep_file1,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name="Separated",
                         line_color=None,
                         ls='--',
                         alpha=1.,
                         single_file=True,
                         incl_label=False)
                plot_mun(axes,
                         mun_comb_con_file1,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name="Contact",
                         line_color=None,
                         ls='-',
                         single_file=True,
                         incl_label=True)
                format_figure(figure,
                              f"Overburden Models\n"
                              f"MUN Plate Contact vs Separation [{conductance} Overburden with Plate 1]",
                              [mun_comb_sep_file1],
                              min_ch,
                              max_ch,
                              incl_legend=True,
                              extra_handles=["--", "-"],
                              extra_labels=["Separated", "Contact"])
                pdf.savefig(figure, orientation='landscape')
                clear_axes(axes)
                log_scale(x_ax_log, z_ax_log)

                plot_mun(axes,
                         mun_comb_sep_file2,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name="Separated",
                         line_color=None,
                         ls='--',
                         alpha=1.,
                         single_file=True,
                         incl_label=False)
                plot_mun(axes,
                         mun_comb_con_file2,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name="Contact",
                         line_color=None,
                         ls='-',
                         single_file=True,
                         incl_label=True)
                format_figure(figure,
                              f"Overburden Models\n"
                              f"MUN Plate Contact vs Separation [{conductance} Overburden with Plate 2]",
                              [mun_comb_sep_file2],
                              min_ch,
                              max_ch,
                              incl_legend=True,
                              extra_handles=["--", "-"],
                              extra_labels=["Separated", "Contact"])
                pdf.savefig(figure, orientation='landscape')
                clear_axes(axes)
                log_scale(x_ax_log, z_ax_log)

            def plot_differential():
                print(F">>Plotting contact differential ({conductance})")
                # Calculate the difference between separate and contact plates
                plot_max(axes,
                         maxwell_plate1_diff,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name="Maxwell",
                         line_color=None,
                         ls=styles.get("Maxwell"),
                         single_file=True,
                         incl_label=True,
                         data_scaling=1e-6,
                         alpha=1.)

                plot_mun(axes,
                         mun_plate1_diff,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name="MUN",
                         line_color=None,
                         ls=styles.get("MUN"),
                         single_file=True,
                         incl_label=False,
                         alpha=0.9)

                format_figure(figure,
                              f"Overburden Models\n"
                              f"Separation vs Contact Differential [{conductance} Overburden with Plate 1]",
                              [maxwell_plate1_diff,
                               mun_plate1_diff],
                              min_ch,
                              max_ch,
                              extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
                              extra_labels=["Maxwell", "MUN"])

                pdf.savefig(figure, orientation='landscape')
                clear_axes(axes)
                log_scale(x_ax_log, z_ax_log)

                plot_max(axes,
                         maxwell_plate2_diff,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name="Maxwell",
                         line_color=None,
                         ls=styles.get("Maxwell"),
                         single_file=True,
                         incl_label=True,
                         data_scaling=1e-6,
                         alpha=1.)

                plot_mun(axes,
                         mun_plate2_diff,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name="MUN",
                         line_color=None,
                         ls=styles.get("MUN"),
                         single_file=True,
                         incl_label=False,
                         alpha=0.9)

                format_figure(figure,
                              f"Overburden Models\n"
                              f"Separation vs Contact Differential [{conductance} Overburden with Plate 2]",
                              [maxwell_plate2_diff, mun_plate2_diff],
                              min_ch,
                              max_ch,
                              extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
                              extra_labels=["Maxwell", "MUN"])

                pdf.savefig(figure, orientation='landscape')
                clear_axes(axes)
                log_scale(x_ax_log, z_ax_log)

            out_pdf = maxwell_folder.parents[1].joinpath(r"Overburden Model - Effects of Plate Contact.PDF")
            with PdfPages(out_pdf) as pdf:

                for conductance in ["1S", "10S"]:
                    maxwell_comb_sep_file1 = TEMFile().parse(
                        Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #1 - 1m Spacing.TEM"))
                    maxwell_comb_sep_file2 = TEMFile().parse(
                        Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #2 - 1m Spacing.TEM"))
                    maxwell_comb_con_file1 = TEMFile().parse(
                        Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #1 - Contact.TEM"))
                    maxwell_comb_con_file2 = TEMFile().parse(
                        Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #2 - Contact.TEM"))

                    mun_comb_sep_file1 = MUNFile().parse(
                        Path(mun_folder).joinpath(fr"{conductance}_overburden_plate250_detach_dBdt.DAT"))
                    mun_comb_sep_file2 = MUNFile().parse(
                        Path(mun_folder).joinpath(fr"{conductance}_overburden_plate50_detach_dBdt.DAT"))
                    mun_comb_con_file1 = MUNFile().parse(
                        Path(mun_folder).joinpath(fr"{conductance}_overburden_plate250_attach_dBdt.DAT"))
                    mun_comb_con_file2 = MUNFile().parse(
                        Path(mun_folder).joinpath(fr"{conductance}_overburden_plate50_attach_dBdt.DAT"))

                    channels = [f'CH{num}' for num in range(1, max_ch - min_ch + 1)]
                    maxwell_plate1_diff = copy.deepcopy(maxwell_comb_sep_file1)
                    maxwell_plate2_diff = copy.deepcopy(maxwell_comb_sep_file2)
                    maxwell_plate1_diff.data.loc[:, channels] = maxwell_comb_con_file1.data.loc[:, channels] - maxwell_comb_sep_file1.data.loc[:, channels]
                    maxwell_plate2_diff.data.loc[:, channels] = maxwell_comb_con_file2.data.loc[:, channels] - maxwell_comb_sep_file2.data.loc[:, channels]

                    mun_plate1_diff = copy.deepcopy(mun_comb_sep_file1)
                    mun_plate2_diff = copy.deepcopy(mun_comb_sep_file2)
                    mun_plate1_diff.data.loc[:, channels] = mun_comb_con_file1.data.loc[:, channels] - mun_comb_sep_file1.data.loc[:, channels]
                    mun_plate2_diff.data.loc[:, channels] = mun_comb_con_file2.data.loc[:, channels] - mun_comb_sep_file2.data.loc[:, channels]

                    plot_maxwell_contact_effect()
                    plot_mun_contact_effect()
                    plot_differential()
            os.startfile(out_pdf)

        def plot_residual(ch_step=1):
            """
            Compare Maxwell and MUN residuals.
            Residual is the effect of mutual induction: combined model - all individual plates.
            """

            def plot_residual_comparison(ch_step=1):
                print(f">> Plotting residual response ({conductance})")

                plot_max(axes,
                         maxwell_plate_1_residual_sep,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name=f"Maxwell",
                         line_color=None,
                         ls=styles.get("Maxwell"),
                         single_file=True,
                         incl_label=True,
                         data_scaling=1e-6,
                         alpha=1.)

                plot_mun(axes,
                         mun_plate_1_residual_sep,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name=f"MUN",
                         line_color=None,
                         ls=styles.get("MUN"),
                         single_file=True,
                         incl_label=False,
                         alpha=1.)

                format_figure(figure,
                              f"Overburden Models\n"
                              f"Residual [{conductance} Overburden with Plate 1, Separated]",
                              [maxwell_plate_1_residual_sep,
                               mun_plate_1_residual_sep],
                              min_ch,
                              max_ch,
                              incl_legend=True,
                              extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
                              extra_labels=["Maxwell", "MUN"])

                pdf.savefig(figure, orientation='landscape')
                clear_axes(axes)
                log_scale(x_ax_log, z_ax_log)

                """Plate 2 with separation"""
                plot_max(axes,
                         maxwell_plate_2_residual_sep,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name=f"Maxwell",
                         line_color=None,
                         ls=styles.get("Maxwell"),
                         single_file=True,
                         incl_label=True,
                         data_scaling=1e-6,
                         alpha=1.)

                plot_mun(axes,
                         mun_plate_2_residual_sep,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name=f"MUN",
                         line_color=None,
                         ls=styles.get("MUN"),
                         single_file=True,
                         incl_label=False,
                         alpha=1.)

                format_figure(figure,
                              f"Overburden Models\nResidual [{conductance} Overburden with Plate 2, Separated]",
                              [maxwell_plate_2_residual_sep,
                               mun_plate_2_residual_sep],
                              min_ch,
                              max_ch,
                              incl_legend=True,
                              extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
                              extra_labels=["Maxwell", "MUN"])

                pdf.savefig(figure, orientation='landscape')
                clear_axes(axes)
                log_scale(x_ax_log, z_ax_log)

                """Plate 1 contact"""
                plot_max(axes,
                         maxwell_plate_1_residual_con,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name=f"Maxwell",
                         line_color=None,
                         ls=styles.get("Maxwell"),
                         single_file=True,
                         incl_label=True,
                         data_scaling=1e-6,
                         alpha=1.)

                plot_mun(axes,
                         mun_plate_1_residual_con,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name=f"MUN",
                         line_color=None,
                         ls=styles.get("MUN"),
                         single_file=True,
                         incl_label=False,
                         alpha=1.)

                format_figure(figure,
                              f"Overburden Models\nResidual [{conductance} Overburden with Plate 1, Contact]",
                              [maxwell_plate_1_residual_con,
                               mun_plate_1_residual_con],
                              min_ch,
                              max_ch,
                              incl_legend=True,
                              extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
                              extra_labels=["Maxwell", "MUN"])

                pdf.savefig(figure, orientation='landscape')
                clear_axes(axes)
                log_scale(x_ax_log, z_ax_log)

                """Plate 2 contact"""
                plot_max(axes,
                         maxwell_plate_2_residual_con,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name=f"Maxwell",
                         line_color=None,
                         ls=styles.get("Maxwell"),
                         single_file=True,
                         incl_label=True,
                         data_scaling=1e-6,
                         alpha=1.)

                plot_mun(axes,
                         mun_plate_2_residual_con,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name=f"MUN",
                         line_color=None,
                         ls=styles.get("MUN"),
                         single_file=True,
                         incl_label=False,
                         alpha=1.)

                format_figure(figure,
                              f"Overburden Models\nResidual [{conductance} Overburden with Plate 2, Contact]",
                              [maxwell_plate_2_residual_con,
                               mun_plate_2_residual_con],
                              min_ch,
                              max_ch,
                              incl_legend=True,
                              extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
                              extra_labels=["Maxwell", "MUN"])

                pdf.savefig(figure, orientation='landscape')
                clear_axes(axes)
                log_scale(x_ax_log, z_ax_log)

            """Compare residual/mutual inductance"""
            out_pdf = maxwell_folder.parents[1].joinpath(r"Overburden Model - Residual.PDF")
            with PdfPages(out_pdf) as pdf:

                for conductance in ["1S", "10S"]:
                    maxwell_ob_file = TEMFile().parse(Path(maxwell_folder).joinpath(fr"{conductance} Overburden Only - 50m.TEM"))
                    mun_ob_file = MUNFile().parse(Path(mun_folder).joinpath(fr"overburden_{conductance}_V1000m_dBdt.DAT"))

                    maxwell_comb_sep_file1 = TEMFile().parse(
                        Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #1 - 1m Spacing.TEM"))
                    maxwell_comb_sep_file2 = TEMFile().parse(
                        Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #2 - 1m Spacing.TEM"))
                    maxwell_comb_con_file1 = TEMFile().parse(
                        Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #1 - Contact.TEM"))
                    maxwell_comb_con_file2 = TEMFile().parse(
                        Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #2 - Contact.TEM"))

                    mun_comb_sep_file1 = MUNFile().parse(
                        Path(mun_folder).joinpath(fr"{conductance}_overburden_plate250_detach_dBdt.DAT"))
                    mun_comb_sep_file2 = MUNFile().parse(
                        Path(mun_folder).joinpath(fr"{conductance}_overburden_plate50_detach_dBdt.DAT"))
                    mun_comb_con_file1 = MUNFile().parse(
                        Path(mun_folder).joinpath(fr"{conductance}_overburden_plate250_attach_dBdt.DAT"))
                    mun_comb_con_file2 = MUNFile().parse(
                        Path(mun_folder).joinpath(fr"{conductance}_overburden_plate50_attach_dBdt.DAT"))

                    maxwell_plate_1_residual_sep = calc_residual(maxwell_comb_sep_file1, maxwell_ob_file, maxwell_plate1_file)
                    maxwell_plate_2_residual_sep = calc_residual(maxwell_comb_sep_file2, maxwell_ob_file, maxwell_plate2_file)
                    maxwell_plate_1_residual_con = calc_residual(maxwell_comb_con_file1, maxwell_ob_file, maxwell_plate1_file)
                    maxwell_plate_2_residual_con = calc_residual(maxwell_comb_con_file2, maxwell_ob_file, maxwell_plate2_file)
                    mun_plate_1_residual_sep = calc_residual(mun_comb_sep_file1, mun_ob_file, mun_plate1_file)
                    mun_plate_2_residual_sep = calc_residual(mun_comb_sep_file2, mun_ob_file, mun_plate2_file)
                    mun_plate_1_residual_con = calc_residual(mun_comb_con_file1, mun_ob_file, mun_plate1_file)
                    mun_plate_2_residual_con = calc_residual(mun_comb_con_file2, mun_ob_file, mun_plate2_file)

                    plot_residual_comparison(ch_step=ch_step)
            os.startfile(out_pdf)

        def analyze_residual(ch_step=1):
            """
            Compare Maxwell and MUN residuals.
            Residual is the effect of mutual induction: combined model - all individual plates.
            """

            def get_residual_diff(maxwell_file, mun_file):
                channels = [f'CH{num}' for num in range(min_ch, max_ch + 1)]
                diff_file = maxwell_file
                diff_data = pd.DataFrame()

                for component in diff_file.data.COMPONENT.unique():
                    maxwell_station_filt = maxwell_file.data.STATION.astype(float).isin(mun_file.data.Station.astype(float) - 0.2)
                    mun_station_filt = (mun_file.data.Station.astype(float) - 0.2).isin(maxwell_file.data.STATION.astype(float))
                    maxwell_filt = (maxwell_file.data.COMPONENT == component) & (maxwell_station_filt)
                    mun_filt = (mun_file.data.Component == component) & (mun_station_filt)

                    maxwell_data = maxwell_file.data[maxwell_filt].reset_index(drop=True).loc[:, channels] * 1e-6
                    mun_data = mun_file.data[mun_filt].reset_index(drop=True).loc[:, channels]

                    diff = maxwell_data.abs() - mun_data.abs()
                    diff.insert(0, "STATION", maxwell_file.data[maxwell_filt].reset_index(drop=True).STATION)
                    diff.insert(0, "COMPONENT", maxwell_file.data[maxwell_filt].reset_index(drop=True).COMPONENT)
                    diff_data = diff_data.append(diff)
                return diff_data

            def get_residual_percent(combined_file, residual_file):
                diff_data = pd.DataFrame()
                for component in combined_file.data.COMPONENT.unique():
                    model_filt = combined_file.data.COMPONENT == component
                    residual_filt = residual_file.data.COMPONENT == component
                    model_data = combined_file.data[model_filt].loc[:, channels].reset_index(drop=True)
                    residual_data = residual_file.data[residual_filt].loc[:, channels].reset_index(drop=True)
                    diff = residual_data / model_data * 100

                    if isinstance(combined_file, TEMFile):
                        diff.insert(0, "STATION", combined_file.data.STATION)
                        diff.insert(0, "COMPONENT", component)
                        diff_data = diff_data.append(diff)
                    else:
                        diff.insert(0, "STATION", combined_file.data.Station)
                        diff.insert(0, "COMPONENT", component)
                        diff_data = diff_data.append(diff)
                return diff_data

            def plot_residual_differential(ch_step=1):
                print(f">> Plotting residual difference ({conductance})")

                """Separated"""
                # Use a maxwell file to make plotting simpler
                # diff_data = get_residual_diff(maxwell_plate_1_residual_sep, mun_plate_1_residual_sep)

                diff_data = get_residual_percent(maxwell_comb_sep_file1, maxwell_plate_1_residual_sep)
                # diff_data = get_residual_percent(mun_comb_sep_file1, mun_plate_1_residual_sep)

                plot_df(axes, diff_data, maxwell_plate_1_residual_sep, ch_step=ch_step)
                # plot_df(axes, diff_data, mun_comb_sep_file1, ch_step=ch_step)
                # plot_df(axes, maxwell_comb_sep_file1.data, mun_comb_sep_file1, ch_step=ch_step,
                #         ls="-")
                # plot_df(axes, maxwell_plate_1_residual_sep.data, mun_comb_sep_file1, ch_step=ch_step,
                #         ls=":")
                format_figure(figure,
                              f"Overburden Models\n"
                              f"Residual Differential [{conductance} Overburden with Plate 1, Separated]",
                              [maxwell_plate_1_residual_sep,
                               mun_plate_1_residual_sep],
                              min_ch,
                              max_ch,
                              incl_legend=True,
                              )

                pdf.savefig(figure, orientation='landscape')
                clear_axes(axes)
                log_scale(x_ax_log, z_ax_log)

                # """Plate 2 with separation"""
                # plot_maxwell(axes,
                #              maxwell_plate_2_residual_sep,
                #              min_ch,
                #              max_ch,
                #              ch_step=ch_step,
                #              name=f"Maxwell",
                #              line_color=None,
                #              line_style=styles.get("Maxwell"),
                #              single_file=True,
                #              incl_label=True,
                #              data_scaling=1e-6,
                #              alpha=1.)
                #
                # plot_mun(axes,
                #          mun_plate_2_residual_sep,
                #          min_ch,
                #          max_ch,
                #          ch_step=ch_step,
                #          name=f"MUN",
                #          line_color=None,
                #          line_style=styles.get("MUN"),
                #          single_file=True,
                #          incl_label=False,
                #          alpha=1.)
                #
                # format_figure(figure,
                #               f"Overburden Models\nResidual [{conductance} Overburden with Plate 2, Separated]",
                #               [maxwell_plate_2_residual_sep,
                #                mun_plate_2_residual_sep],
                #               min_ch,
                #               max_ch,
                #               incl_legend=True,
                #               extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
                #               extra_labels=["Maxwell", "MUN"])
                #
                # pdf.savefig(figure, orientation='landscape')
                # clear_axes(axes)
                # log_scale(x_ax_log, z_ax_log)
                #
                # """Plate 1 contact"""
                # plot_maxwell(axes,
                #              maxwell_plate_1_residual_con,
                #              min_ch,
                #              max_ch,
                #              ch_step=ch_step,
                #              name=f"Maxwell",
                #              line_color=None,
                #              line_style=styles.get("Maxwell"),
                #              single_file=True,
                #              incl_label=True,
                #              data_scaling=1e-6,
                #              alpha=1.)
                #
                # plot_mun(axes,
                #          mun_plate_1_residual_con,
                #          min_ch,
                #          max_ch,
                #          ch_step=ch_step,
                #          name=f"MUN",
                #          line_color=None,
                #          line_style=styles.get("MUN"),
                #          single_file=True,
                #          incl_label=False,
                #          alpha=1.)
                #
                # format_figure(figure,
                #               f"Overburden Models\nResidual [{conductance} Overburden with Plate 1, Contact]",
                #               [maxwell_plate_1_residual_con,
                #                mun_plate_1_residual_con],
                #               min_ch,
                #               max_ch,
                #               incl_legend=True,
                #               extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
                #               extra_labels=["Maxwell", "MUN"])
                #
                # pdf.savefig(figure, orientation='landscape')
                # clear_axes(axes)
                # log_scale(x_ax_log, z_ax_log)
                #
                # """Plate 2 contact"""
                # plot_maxwell(axes,
                #              maxwell_plate_2_residual_con,
                #              min_ch,
                #              max_ch,
                #              ch_step=ch_step,
                #              name=f"Maxwell",
                #              line_color=None,
                #              line_style=styles.get("Maxwell"),
                #              single_file=True,
                #              incl_label=True,
                #              data_scaling=1e-6,
                #              alpha=1.)
                #
                # plot_mun(axes,
                #          mun_plate_2_residual_con,
                #          min_ch,
                #          max_ch,
                #          ch_step=ch_step,
                #          name=f"MUN",
                #          line_color=None,
                #          line_style=styles.get("MUN"),
                #          single_file=True,
                #          incl_label=False,
                #          alpha=1.)
                #
                # format_figure(figure,
                #               f"Overburden Models\nResidual [{conductance} Overburden with Plate 2, Contact]",
                #               [maxwell_plate_2_residual_con,
                #                mun_plate_2_residual_con],
                #               min_ch,
                #               max_ch,
                #               incl_legend=True,
                #               extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
                #               extra_labels=["Maxwell", "MUN"])
                #
                # pdf.savefig(figure, orientation='landscape')
                # clear_axes(axes)
                # log_scale(x_ax_log, z_ax_log)

            """Compare residual/mutual inductance"""
            out_pdf = maxwell_folder.parents[1].joinpath(r"Overburden Model - Residual Analysis.PDF")
            with PdfPages(out_pdf) as pdf:

                for conductance in ["1S", "10S"]:
                    maxwell_ob_file = TEMFile().parse(Path(maxwell_folder).joinpath(fr"{conductance} Overburden Only - 50m.TEM"))
                    mun_ob_file = MUNFile().parse(Path(mun_folder).joinpath(fr"overburden_{conductance}_V1000m_dBdt.DAT"))

                    maxwell_comb_sep_file1 = TEMFile().parse(
                        Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #1 - 1m Spacing.TEM"))
                    maxwell_comb_sep_file2 = TEMFile().parse(
                        Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #2 - 1m Spacing.TEM"))
                    maxwell_comb_con_file1 = TEMFile().parse(
                        Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #1 - Contact.TEM"))
                    maxwell_comb_con_file2 = TEMFile().parse(
                        Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #2 - Contact.TEM"))

                    mun_comb_sep_file1 = MUNFile().parse(
                        Path(mun_folder).joinpath(fr"{conductance}_overburden_plate250_detach_dBdt.DAT"))
                    mun_comb_sep_file2 = MUNFile().parse(
                        Path(mun_folder).joinpath(fr"{conductance}_overburden_plate50_detach_dBdt.DAT"))
                    mun_comb_con_file1 = MUNFile().parse(
                        Path(mun_folder).joinpath(fr"{conductance}_overburden_plate250_attach_dBdt.DAT"))
                    mun_comb_con_file2 = MUNFile().parse(
                        Path(mun_folder).joinpath(fr"{conductance}_overburden_plate50_attach_dBdt.DAT"))

                    maxwell_plate_1_residual_sep = calc_residual(maxwell_comb_sep_file1, maxwell_ob_file, maxwell_plate1_file)
                    maxwell_plate_2_residual_sep = calc_residual(maxwell_comb_sep_file2, maxwell_ob_file, maxwell_plate2_file)
                    maxwell_plate_1_residual_con = calc_residual(maxwell_comb_con_file1, maxwell_ob_file, maxwell_plate1_file)
                    maxwell_plate_2_residual_con = calc_residual(maxwell_comb_con_file2, maxwell_ob_file, maxwell_plate2_file)
                    mun_plate_1_residual_sep = calc_residual(mun_comb_sep_file1, mun_ob_file, mun_plate1_file)
                    mun_plate_2_residual_sep = calc_residual(mun_comb_sep_file2, mun_ob_file, mun_plate2_file)
                    mun_plate_1_residual_con = calc_residual(mun_comb_con_file1, mun_ob_file, mun_plate1_file)
                    mun_plate_2_residual_con = calc_residual(mun_comb_con_file2, mun_ob_file, mun_plate2_file)

                    plot_residual_differential(ch_step=ch_step)
            os.startfile(out_pdf)

        def plot_enhancement(ch_step=1):
            """
            Compare Maxwell and MUN plate enhancement
            """

            def calc_enhancement(combined_file, ob_file, plate_file):
                # Works for both MUN and Maxwell
                print(f"Calculating enhancement for {', '.join([f.filepath.name for f in [combined_file, ob_file, plate_file]])}")
                enhance_file = copy.deepcopy(plate_file)
                channels = [f'CH{num}' for num in range(1, len(ob_file.ch_times) + 1)]

                enhance_data = combined_file.data.loc[:, channels] - ob_file.data.loc[:, channels]
                enhance_file.data.loc[:, channels] = enhance_data
                if isinstance(plate_file, TEMFile):
                    global count
                    if count == 0 or count == 1:
                        sep = "Separated"
                    else:
                        sep = "Contact"
                    filepath = enhance_file.filepath.parent.with_name(enhance_file.filepath.stem +
                                                                      f" ({sep}, {conductance} overburden enhancement).TEM")
                    count += 1
                    enhance_file.save(filepath=filepath)
                return enhance_file

            def plot_enhancement_comparison(ch_step=1):
                """Compare plate enhancement"""
                plot_max(axes,
                         maxwell_plate_1_enhance_sep,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name=f"Maxwell",
                         line_color=None,
                         ls=styles.get("Maxwell"),
                         single_file=True,
                         incl_label=True,
                         data_scaling=1e-6,
                         alpha=1.)

                plot_mun(axes,
                         mun_plate_1_enhance_sep,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name=f"MUN",
                         line_color=None,
                         ls=styles.get("MUN"),
                         single_file=True,
                         incl_label=False,
                         alpha=1.)

                format_figure(figure,
                              f"Overburden Models\n"
                              f"Plate Enhancement (Overburden Response Substracted) [{conductance} Overburden with Plate 1, Separated]",
                              [maxwell_plate_1_enhance_sep,
                               mun_plate_1_enhance_sep],
                              min_ch,
                              max_ch,
                              incl_legend=True,
                              extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
                              extra_labels=["Maxwell", "MUN"])

                pdf.savefig(figure, orientation='landscape')
                clear_axes(axes)
                log_scale(x_ax_log, z_ax_log)

                """Plate 2 with separation"""
                plot_max(axes,
                         maxwell_plate_2_enhance_sep,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name=f"Maxwell",
                         line_color=None,
                         ls=styles.get("Maxwell"),
                         single_file=True,
                         incl_label=True,
                         data_scaling=1e-6,
                         alpha=1.)

                plot_mun(axes,
                         mun_plate_2_enhance_sep,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name=f"MUN",
                         line_color=None,
                         ls=styles.get("MUN"),
                         single_file=True,
                         incl_label=False,
                         alpha=1.)

                format_figure(figure,
                              f"Overburden Models\n"
                              f"Plate Enhancement (Overburden Response Substracted) [{conductance} Overburden with Plate 2, Separated]",
                              [maxwell_plate_2_enhance_sep,
                               mun_plate_2_enhance_sep],
                              min_ch,
                              max_ch,
                              incl_legend=True,
                              extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
                              extra_labels=["Maxwell", "MUN"])

                pdf.savefig(figure, orientation='landscape')
                clear_axes(axes)
                log_scale(x_ax_log, z_ax_log)

                """Plate 1 contact"""
                plot_max(axes,
                         maxwell_plate_1_enhance_con,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name=f"Maxwell",
                         line_color=None,
                         ls=styles.get("Maxwell"),
                         single_file=True,
                         incl_label=True,
                         data_scaling=1e-6,
                         alpha=1.)

                plot_mun(axes,
                         mun_plate_1_enhance_con,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name=f"MUN",
                         line_color=None,
                         ls=styles.get("MUN"),
                         single_file=True,
                         incl_label=False,
                         alpha=1.)

                format_figure(figure,
                              f"Overburden Models\n"
                              f"Plate Enhancement (Overburden Response Substracted) [{conductance} Overburden with Plate 1, Contact]",
                              [maxwell_plate_1_enhance_con,
                               mun_plate_1_enhance_con],
                              min_ch,
                              max_ch,
                              incl_legend=True,
                              extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
                              extra_labels=["Maxwell", "MUN"])

                pdf.savefig(figure, orientation='landscape')
                clear_axes(axes)
                log_scale(x_ax_log, z_ax_log)

                """Plate 2 contact"""
                plot_max(axes,
                         maxwell_plate_2_enhance_con,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name=f"Maxwell",
                         line_color=None,
                         ls=styles.get("Maxwell"),
                         single_file=True,
                         incl_label=True,
                         data_scaling=1e-6,
                         alpha=1.)

                plot_mun(axes,
                         mun_plate_2_enhance_con,
                         min_ch,
                         max_ch,
                         ch_step=ch_step,
                         name=f"MUN",
                         line_color=None,
                         ls=styles.get("MUN"),
                         single_file=True,
                         incl_label=False,
                         alpha=1.)

                format_figure(figure,
                              f"Overburden Models\n"
                              f"Plate Enhancement (Overburden Response Substracted) [{conductance} Overburden with Plate 2, Contact]",
                              [maxwell_plate_2_enhance_con,
                               mun_plate_2_enhance_con],
                              min_ch,
                              max_ch,
                              incl_legend=True,
                              extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
                              extra_labels=["Maxwell", "MUN"])

                pdf.savefig(figure, orientation='landscape')
                clear_axes(axes)
                log_scale(x_ax_log, z_ax_log)

            out_pdf = maxwell_folder.parents[1].joinpath(r"Overburden Model - Enhancement.PDF")
            with PdfPages(out_pdf) as pdf:

                for conductance in ["1S", "10S"]:
                    print(f">> Plotting enhancement ({conductance})")
                    maxwell_ob_file = TEMFile().parse(Path(maxwell_folder).joinpath(fr"{conductance} Overburden Only - 50m.TEM"))
                    mun_ob_file = MUNFile().parse(Path(mun_folder).joinpath(fr"overburden_{conductance}_V1000m_dBdt.DAT"))

                    maxwell_comb_sep_file1 = TEMFile().parse(
                        Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #1 - 1m Spacing.TEM"))
                    maxwell_comb_sep_file2 = TEMFile().parse(
                        Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #2 - 1m Spacing.TEM"))
                    maxwell_comb_con_file1 = TEMFile().parse(
                        Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #1 - Contact.TEM"))
                    maxwell_comb_con_file2 = TEMFile().parse(
                        Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #2 - Contact.TEM"))

                    mun_comb_sep_file1 = MUNFile().parse(
                        Path(mun_folder).joinpath(fr"{conductance}_overburden_plate250_detach_dBdt.DAT"))
                    mun_comb_sep_file2 = MUNFile().parse(
                        Path(mun_folder).joinpath(fr"{conductance}_overburden_plate50_detach_dBdt.DAT"))
                    mun_comb_con_file1 = MUNFile().parse(
                        Path(mun_folder).joinpath(fr"{conductance}_overburden_plate250_attach_dBdt.DAT"))
                    mun_comb_con_file2 = MUNFile().parse(
                        Path(mun_folder).joinpath(fr"{conductance}_overburden_plate50_attach_dBdt.DAT"))

                    global count
                    count = 0
                    maxwell_plate_1_enhance_sep = calc_enhancement(maxwell_comb_sep_file1, maxwell_ob_file, maxwell_plate1_file)
                    maxwell_plate_2_enhance_sep = calc_enhancement(maxwell_comb_sep_file2, maxwell_ob_file, maxwell_plate2_file)
                    maxwell_plate_1_enhance_con = calc_enhancement(maxwell_comb_con_file1, maxwell_ob_file, maxwell_plate1_file)
                    maxwell_plate_2_enhance_con = calc_enhancement(maxwell_comb_con_file2, maxwell_ob_file, maxwell_plate2_file)
                    mun_plate_1_enhance_sep = calc_enhancement(mun_comb_sep_file1, mun_ob_file, mun_plate1_file)
                    mun_plate_2_enhance_sep = calc_enhancement(mun_comb_sep_file2, mun_ob_file, mun_plate2_file)
                    mun_plate_1_enhance_con = calc_enhancement(mun_comb_con_file1, mun_ob_file, mun_plate1_file)
                    mun_plate_2_enhance_con = calc_enhancement(mun_comb_con_file2, mun_ob_file, mun_plate2_file)

                    # plot_enhancement_comparison(ch_step=ch_step)
            # os.startfile(out_pdf)

        figure, ((x_ax, x_ax_log), (z_ax, z_ax_log)) = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='col')
        axes = [x_ax, z_ax, x_ax_log, z_ax_log]
        figure.set_size_inches((11 * 1.33, 8.5 * 1.33))
        log_scale(x_ax_log, z_ax_log)

        maxwell_folder = sample_files.joinpath(r"Overburden\Maxwell\Overburden+Conductor Revised")
        assert maxwell_folder.is_dir(), f"{maxwell_folder} is not a directory."
        mun_folder = sample_files.joinpath(r"Overburden\MUN\Overburden + plate")
        assert mun_folder.is_dir(), f"{mun_folder} is not a directory."

        global min_ch, max_ch, channels, channel_step
        min_ch, max_ch = 21, 44
        channels = [f"CH{num}" for num in range(min_ch, max_ch + 1)]
        channel_step = 1

        maxwell_plate1_file = TEMFile().parse(Path(maxwell_folder).joinpath(r"Plate #1 Only - 51m.TEM"))
        maxwell_plate2_file = TEMFile().parse(Path(maxwell_folder).joinpath(r"Plate #2 Only - 51m.TEM"))
        mun_plate1_file = MUNFile().parse(Path(mun_folder).joinpath(r"only_plate250_dBdt.DAT"))
        mun_plate2_file = MUNFile().parse(Path(mun_folder).joinpath(r"only_plate50_dBdt.DAT"))

        # plot_overburden_and_plates(ch_step=channel_step)
        # plot_contact_effect(ch_step=channel_step)
        # plot_residual(ch_step=channel_step)
        analyze_residual(ch_step=channel_step)
        # plot_enhancement(ch_step=channel_step)

        print(F"Plotting complete.")

    def plot_bentplate():

        def plot_model(model_name, title, pdf, max_folder, mun_folder, residual=False, y_min=None, y_max=None, ylabel=''):
            print(f"Searching for {model_name}.TEM")
            max_file = max_folder.joinpath(model_name).with_suffix(".TEM")
            mun_file = mun_folder.joinpath(model_name).with_suffix(".DAT")
            file_times = None
            files = []

            # if not any([max_file.is_file(), mun_file.is_file()]):
            #     print(F"Model {model_name} not found for any files.")
            #     logging_file.write(F"Model {model_name} not found for any files.\n")
            #     return
            # else:
            #     if max_file.is_file():
            #         print(f"Plotting {max_file.name}.")
            #         tem_file = TEMFile().parse(max_file)
            #         if residual is True:
            #             tem_file = get_residual_file(tem_file, max_folder)
            #         files.append(tem_file)
            #         log_scale(x_ax_log, z_ax_log)
            #         plot_max(axes, tem_file, min_ch, max_ch,
            #                  ch_step=channel_step,
            #                  name=max_file.name,
            #                  ls=styles.get("Maxwell"),
            #                  station_shift=-200,
            #                  data_scaling=1e-6,
            #                  y_min=y_min,
            #                  y_max=y_max,
            #                  alpha=0.5)
            #         file_times = tem_file.ch_times
            #     else:
            #         print(F"Maxwell file {max_file.name} not found.")
            #         logging_file.write(F"Maxwell file {max_file.name} not found.\n")

            if mun_file.is_file():
                print(f"Plotting {mun_file.name}.")
                dat_file = MUNFile().parse(mun_file)
                if residual is True:
                    dat_file = get_residual_file(dat_file, mun_folder, single_plot_order)
                files.append(dat_file)
                log_scale(x_ax_log, z_ax_log)
                plot_mun(axes, dat_file, min_ch, max_ch,
                         ch_step=channel_step,
                         name=mun_file.name,
                         ls=styles.get("MUN"),
                         station_shift=0,
                         data_scaling=1.,
                         y_min=y_min,
                         y_max=y_max)
                if file_times is None:
                    file_times = dat_file.ch_times
            else:
                print(F"MUN file {mun_file.name} not found.")
                logging_file.write(F"MUN file {mun_file.name} not found.\n")

            name = "Multiple and Bent Plate Models\n" + title + " " + model_name
            format_figure(figure, name, files, min_ch, max_ch,
                          ch_step=channel_step,
                          b_field=False,
                          incl_legend=True,
                          incl_legend_ls=True,
                          legend_times=file_times,
                          ylabel=ylabel)
            pdf.savefig(figure, orientation='landscape')
            clear_axes(axes)

        def plot_individual_plates(fixed_y=False):
            """ Plot individual plates"""
            out_pdf = sample_files.joinpath(
                r"Bent and Multiple Plates\Multiple and Bent Plate Models - Individual Plates.PDF")
            if fixed_y is True:
                y_min, y_max = mn, mx
            else:
                y_min, y_max = None, None

            logging_file.write(f">>Plotting Multiple and Bent Plates - Individual Plates<<\n")
            print(f">>Plotting individual plates")
            count = 0
            with PdfPages(out_pdf) as pdf:
                for model in single_plot_order:
                    print(f"Plotting model {model} ({count + 1}/{len(single_plot_order)})")
                    plot_model(model, "Individual Plate:", pdf, max_folder_100S, mun_folder_100S, y_min=y_min, y_max=y_max)
                    count += 1
            os.startfile(out_pdf)

        def plot_combined_plates(fixed_y=False):
            """ Plot combined plates"""
            out_pdf = sample_files.joinpath(
                r"Bent and Multiple Plates\Multiple and Bent Plate Models - Combined Plates.PDF")
            if fixed_y is True:
                y_min, y_max = mn, mx
            else:
                y_min, y_max = None, None

            logging_file.write(f">>Plotting Multiple and Bent Plates - Combined Plots<<\n")
            print(f">>Plotting combined plates")
            count = 0
            with PdfPages(out_pdf) as pdf:
                for model in combined_plot_order:
                    print(f"Plotting model {model} ({count + 1}/{len(combined_plot_order)})")
                    plot_model(model, "Combined Plates:", pdf, max_folder_100S, mun_folder_100S, y_min=y_min, y_max=y_max)
                    count += 1
            os.startfile(out_pdf)

        def plot_residual(fixed_y=False):
            """ Plot residuals """
            if fixed_y is True:
                y_min, y_max = mn, mx
            else:
                y_max, y_min = None, None

            out_pdf = sample_files.joinpath(
                r"Bent and Multiple Plates\Multiple and Bent Plate Models - Residual Calculation.PDF")
            with PdfPages(out_pdf) as pdf:
                logging_file.write(f">>Plotting Multiple and Bent Plates - Residual Calculation<<\n")
                count = 0
                print(f">>Plotting plate residuals")

                combined_files = [f for f in combined_plot_order if len(f) > 1]
                for model in combined_files:
                    plot_model(model, "Residual:", pdf, max_folder_100S, mun_folder_100S,
                               residual=True,
                               y_min=y_min,
                               y_max=y_max,
                               ylabel="Residual (nT/s)")
                    count += 1

            os.startfile(out_pdf)

        def plot_varying_conductances(fixed_y=False):
            """ Plot various conductances """
            models = {
                "1@10S": "1",
                "1@1000S": "1",
                "4@1000S": "4",
                "1@10S_4@100S": "1_4",
                "1@10S_4@1000S": "1_4",
                "1@100S_4@1000S": "1_4",
                "1@10S+4@100S": "1_4",
                "1@10S+4@1000S": "1_4",
                "1@100S+4@1000S": "1_4",
                "1@10S_2@100S": "1_2",
                "1@10S+2@100S": "1+2",
            }

            out_pdf = sample_files.joinpath(
                r"Bent and Multiple Plates\Multiple and Bent Plate Models - Various Conductances.PDF")

            if fixed_y is True:
                y_min, y_max = mn, mx
            else:
                y_min, y_max = None, None

            logging_file.write(f">>Plotting Multiple and Bent Plates - Various Conductances<<\n")
            print(f">>Plotting various conductance plates.")
            count = 0
            with PdfPages(out_pdf) as pdf:
                for model in models.keys():
                    print(f"Plotting model {model} ({count + 1}/{len(models)})")
                    plot_model(model, "Various Conductances:", pdf, max_folder_varying, mun_folder_varying, y_min=y_min,
                               y_max=y_max)
                    count += 1
            os.startfile(out_pdf)

            # pair_file = folder_100S.joinpath(models.get(model)).with_suffix(".TEM")
            # if not pair_file.is_file():
            #     print(f"Could not find {pair_file.name}.")
            #     logging_file.write(f"Could not find {pair_file.name}.\n")
            # else:
            #     pair_tem = TEMFile().parse(pair_file)
            #     plot_maxwell(axes, pair_tem, "#cf0029", min_ch, max_ch, ch_step=channel_step, line_style='-',
            #                  name=pair_file.name, station_shift=-200, data_scaling=1e-6,
            #                  x_min=-200, x_max=400, y_min=y_min, y_max=y_max, alpha=0.9)

        max_folder_100S = sample_files.joinpath(r"Bent and Multiple Plates\Maxwell\Revised\100S Plates")
        mun_folder_100S = sample_files.joinpath(r"Bent and Multiple Plates\MUN\100S Plates")
        max_folder_varying = sample_files.joinpath(r"Bent and Multiple Plates\Maxwell\Revised\Various Conductances")
        mun_folder_varying = sample_files.joinpath(r"Bent and Multiple Plates\MUN\Various Conductances")
        assert all([max_folder_100S.exists(), mun_folder_100S.exists(), max_folder_varying.exists(), mun_folder_varying.exists()]), \
            "One or more of the folders doesn't exist."
        figure, ((x_ax, x_ax_log), (z_ax, z_ax_log)) = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='col')
        axes = [x_ax, z_ax, x_ax_log, z_ax_log]
        figure.set_size_inches((11 * 1.33, 8.5 * 1.33))

        global min_ch, max_ch, channel_step
        min_ch, max_ch = 21, 21
        channel_step = 1

        single_plot_order = [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            ]

        combined_plot_order = [
            "1_2",
            "1+2",
            "1_4",
            "1+4",
            "2_3",
            "2+3",
            "2_4",
            "2+4",
            "2_5",
            "2+5",
            "4_5",
            "4+5",
            "5_3",
            "5+3",
            "3_6",
            "3+6",
            "1_4_5",
            "1+4+5",
            "5_3_6",
            "5+3+6",
            "1_2_3",
            "1+2+3",
            "1_2_3_6",
            "1+2+3+6",
            "(1+2)_(3+6)",
            "(1+2+3)_6",
            "1_4_5_3_6",
            "1+4+5+3+6",
            "(1+4+5)_(3+6)",
            ]

        mn, mx = get_folder_range(max_folder_100S, "Maxwell", start_ch=min_ch, end_ch=max_ch)
        mn, mx = mn * 1e-6, mx * 1e-6
        # plot_individual_plates(fixed_y=False)
        # plot_combined_plates(fixed_y=False)
        plot_residual(fixed_y=False)

        # Varying conductances
        # mn, mx = get_folder_range(folder_varying, "Maxwell", start_ch=min_ch, end_ch=max_ch)
        # mn, mx = mn * 1e-6, mx * 1e-6
        # plot_varying_conductances(fixed_y=False)

    def test_savgol_filter():

        def format_figure(figure, title, files, min_ch, max_ch, ylabel=''):
            for legend in figure.legends:
                legend.remove()

            rainbow_colors = cm.jet(np.linspace(0, 1, (max_ch - min_ch) + 1))
            x_ax, x_ax_log, z_ax, z_ax_log = figure.axes
            # Set the labels
            z_ax.set_xlabel(f"Station")
            z_ax_log.set_xlabel(f"Station")
            if ylabel:
                for ax in figure.axes:
                    ax.set_ylabel(ylabel)
            else:
                for ax in figure.axes:
                    ax.set_ylabel(f"EM Response\n(nT/s)")

            figure.suptitle(title)
            x_ax.set_title(f"X Component")
            z_ax.set_title(f"Z Component")
            x_ax_log.set_title(f"X Component")
            z_ax_log.set_title(f"Z Component")

            # Create a legend from the plotted lines
            lines, times = [], []
            for i, ch in enumerate(range(min_ch, max_ch + 1)):
                line = Line2D([0], [0], color=rainbow_colors[i], linestyle="-")
                lines.append(line)
                times.append(f"{files[0].ch_times[ch]:.3f}ms")
            handles = lines
            labels = times
            ax_handles, ax_labels = z_ax.get_legend_handles_labels()
            handles.extend(ax_handles)
            labels.extend(ax_labels)

            figure.legend(handles, labels, loc='upper right')

        def plot_model(model_name, title, pdf, mun_folder, y_min=None, y_max=None, ylabel=''):
            print(f"Searching for {model_name}.TEM")
            mun_file = mun_folder.joinpath(model_name).with_suffix(".DAT")
            file_times = None
            files = []

            if mun_file.is_file():
                print(f"Plotting {mun_file.name}.")
                dat_file = MUNFile().parse(mun_file)
                residual_file = get_residual_file(dat_file, mun_folder, single_plot_order)
                files.append(residual_file)

                channel_tuples = list(zip(list(range(min_ch, max_ch + 1))[:-num_chs - 1: num_chs],
                                          list(range(min_ch, max_ch + 1))[num_chs - 1:: num_chs]))
                for chs in channel_tuples:
                    print(F"Plotting channel {chs[0]}-{chs[-1]}.")
                    # Plot the normal residual
                    log_scale(x_ax_log, z_ax_log)
                    plot_mun(axes, residual_file, chs[0], chs[-1],
                             name="Original",
                             ls=":",
                             station_shift=0,
                             data_scaling=1.,
                             y_min=y_min,
                             y_max=y_max,
                             filter=False)

                    # Plot the filtered residual
                    log_scale(x_ax_log, z_ax_log)
                    plot_mun(axes, residual_file, chs[0], chs[-1],
                             name="Filtered",
                             ls="-",
                             station_shift=0,
                             data_scaling=1.,
                             y_min=y_min,
                             y_max=y_max,
                             filter=True)

                    if file_times is None:
                        file_times = dat_file.ch_times

                    name = f"Savitzky–Golay Filter\nModel {title}, Channel {chs[0]}-{chs[-1]}"
                    format_figure(figure, name, files, chs[0], chs[-1], ylabel=ylabel)
                    pdf.savefig(figure, orientation='landscape')
                    clear_axes(axes)
            else:
                print(F"MUN file {mun_file.name} not found.")
                logging_file.write(F"MUN file {mun_file.name} not found.\n")

        mun_folder_100S = sample_files.joinpath(r"Bent and Multiple Plates\MUN\100S Plates")
        mun_folder_varying = sample_files.joinpath(r"Bent and Multiple Plates\MUN\Various Conductances")
        assert all([mun_folder_100S.exists(), mun_folder_varying.exists()]), "One or more of the folders doesn't exist."
        figure, ((x_ax, x_ax_log), (z_ax, z_ax_log)) = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='col')
        axes = [x_ax, z_ax, x_ax_log, z_ax_log]
        figure.set_size_inches((11 * 1.33, 8.5 * 1.33))

        global min_ch, max_ch
        min_ch, max_ch = 21, 44
        num_chs = 3

        single_plot_order = [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
        ]

        combined_plot_order = [
            "1_2",
            "1+2",
            "1_4",
            "1+4",
            "2_3",
            "2+3",
            "2_4",
            "2+4",
            "2_5",
            "2+5",
            "4_5",
            "4+5",
            "5_3",
            "5+3",
            "3_6",
            "3+6",
            "1_4_5",
            "1+4+5",
            "5_3_6",
            "5+3+6",
            "1_2_3",
            "1+2+3",
            "1_2_3_6",
            "1+2+3+6",
            "(1+2)_(3+6)",
            "(1+2+3)_6",
            "1_4_5_3_6",
            "1+4+5+3+6",
            "(1+4+5)_(3+6)",
        ]

        """ Plot residuals """
        out_pdf = sample_files.joinpath(
            r"Bent and Multiple Plates\Savitzky-Golay Filter.PDF")
        with PdfPages(out_pdf) as pdf:
            logging_file.write(f">>Savgol Filter Testing<<\n")
            count = 0
            print(f">>Savgol filter test")

            combined_files = [f for f in combined_plot_order if len(f) > 1][:2]
            for model in combined_files:
                plot_model(model, model, pdf, mun_folder_100S,
                           ylabel="Residual Response (nT/s)")
                count += 1

        os.startfile(out_pdf)


    # TODO Change "MUN" to "EM3D"
    plot_aspect_ratio()
    # plot_two_way_induction()
    # plot_run_on_comparison()
    # plot_run_on_convergence()
    # tabulate_run_on_convergence()
    # compare_maxwell_ribbons()
    # compare_step_on_b_with_theory()
    # plot_overburden()
    # plot_bentplate()
    # test_savgol_filter()

    # tester = TestRunner()
    # tester.show()
    # tester.add_row(sample_files.joinpath(r"Two-way induction\300x100\100S\MUN"), file_type="MUN")
    # # tester.add_row(sample_files.joinpath(r"Two-way induction\300x100\100S\Maxwell"), file_type="Maxwell")
    # tester.test_name_edit.setText("Testing this bullshit")
    # tester.output_filepath_edit.setText(str(sample_files.joinpath(
    #     r"Two-way induction\300x100\100S\MUN\MUN plotting test.PDF")))
    # tester.print_pdf()

    logging_file.close()
    # os.startfile(log_file_path)
    app.exec_()
