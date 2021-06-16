import copy
import io
import math
import os
import pickle
import re
import sys
from itertools import zip_longest
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5 import (QtCore, QtGui, uic)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMessageBox, QFrame, QErrorMessage, QFileDialog,
                             QTableWidgetItem, QScrollArea, QSpinBox, QHBoxLayout, QLabel, QInputDialog, QLineEdit,
                             QProgressDialog, QWidget, QHeaderView, QPushButton, QColorDialog)
from cycler import cycler
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.pyplot import cm
from matplotlib.ticker import MaxNLocator
from natsort import natsorted, os_sorted
from scipy.signal import savgol_filter
from scipy import interpolate

from src.file_types.fem_file import FEMTab
from src.file_types.irap_file import IRAPFile
from src.file_types.mun_file import MUNFile, MUNTab
from src.file_types.platef_file import PlateFFile, PlateFTab
from src.file_types.tem_file import TEMFile, TEMTab

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
# matplotlib.rc('lines', color='gray')

rainbow_colors = iter(cm.rainbow(np.linspace(0, 1, 20)))
quant_colors = np.nditer(np.array(plt.rcParams['axes.prop_cycle'].by_key()['color']))

# iter_colors = np.nditer(quant_colors)
# quant_colors = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
# color = iter(cm.tab10())

extensions = {"Maxwell": "*.TEM", "MUN": "*.DAT", "IRAP": "*.DAT", "PLATE": "*.DAT"}
colors = {"Maxwell": '#0000FF', "MUN": '#43cc31', "IRAP": "#000000", "PLATE": '#FF0000'}
styles = {"Maxwell": "-", "MUN": ":", "IRAP": "--", "PLATE": '-.'}


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
            del extensions[type]
            print(f"New options: {extensions}")

        # Don't add any  more rows if all file types have been selected
        if len(extensions) == 0:
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

    def plot_obj(ax_dict, file, ch_start, ch_end, ch_step=1, ch_times=None, name="", station_shift=0,
                 data_scaling=1., alpha=1., ls=None, lc=None, filter=False):

        x_ax, x_ax_log = ax_dict.get('X')
        y_ax, y_ax_log = ax_dict.get('Y')
        z_ax, z_ax_log = ax_dict.get('Z')
        axes = [x_ax, x_ax_log, y_ax, y_ax_log, z_ax, z_ax_log]

        rainbow_colors = cm.jet(np.linspace(0, ch_step, (ch_end - ch_start) + 1))
        line_styles = ['-', '--', '-.', ':']

        for ax in axes:
            if ax:
                if lc is None:
                    ax.set_prop_cycle(cycler("color", rainbow_colors))
                else:
                    ax.set_prop_cycle(cycler("linestyle", line_styles))

        if isinstance(file, TEMFile):
            x_data = file.data[(file.data.COMPONENT == "X") | (file.data.COMPONENT == "U")]
            y_data = file.data[(file.data.COMPONENT == "Y") | (file.data.COMPONENT == "V")]
            z_data = file.data[(file.data.COMPONENT == "Z") | (file.data.COMPONENT == "A")]
            channels = [f'CH{num}' for num in range(1, len(file.ch_times) + 1)]
        elif isinstance(file, MUNFile):
            x_data = file.data[(file.data.Component == "X") | (file.data.Component == "U")]
            y_data = file.data[(file.data.Component == "Y") | (file.data.Component == "V")]
            z_data = file.data[(file.data.Component == "Z") | (file.data.Component == "A")]
            channels = [f'CH{num}' for num in range(1, len(file.ch_times) + 1)]
        elif isinstance(file, PlateFFile):
            x_data = file.data[(file.data.Component == "X") | (file.data.Component == "U")]
            y_data = file.data[(file.data.Component == "Y") | (file.data.Component == "V")]
            z_data = file.data[(file.data.Component == "Z") | (file.data.Component == "A")]
            channels = [f'{num}' for num in range(1, len(file.ch_times) + 1)]
        elif isinstance(file, IRAPFile):
            x_data = file.data[(file.data.Component == "X") | (file.data.Component == "U")]
            y_data = file.data[(file.data.Component == "Y") | (file.data.Component == "V")]
            z_data = file.data[(file.data.Component == "Z") | (file.data.Component == "A")]
            channels = [f'{num}' for num in range(1, len(file.ch_times) + 1)]
        elif isinstance(file, pd.DataFrame):
            if ch_times is None:
                raise ValueError(f"ch_times cannot be None if a DataFrame is passed.")
            x_data = file.data[(file.data.Component == "X") | (file.data.Component == "U")]
            y_data = file.data[(file.data.Component == "Y") | (file.data.Component == "V")]
            z_data = file.data[(file.data.Component == "Z") | (file.data.Component == "A")]
            channels = [f'CH{num}' for num in range(1, len(ch_times) + 1)]
        else:
            raise ValueError(f"{file} is not a valid input type.")

        min_ch = ch_start - 1
        max_ch = min(ch_end - 1, len(channels) - 1)
        plotting_channels = channels[min_ch: max_ch + 1: ch_step]
        if ch_end > len(channels):
            raise ValueError(f"Channel {ch_end} is beyond the number of channels ({len(channels)}).")

        for ind, ch in enumerate(plotting_channels):
            if ind == 0:
                if not name:
                    name = get_filetype(file)
                label = name
            else:
                label = None

            if isinstance(file, TEMFile):
                x = z_data.STATION.astype(float) + station_shift
            else:
                x = z_data.Station.astype(float) + station_shift

            xx = x_data.loc[:, ch].astype(float) * data_scaling
            yy = y_data.loc[:, ch].astype(float) * data_scaling
            zz = z_data.loc[:, ch].astype(float) * data_scaling

            if filter is True:
                xx = savgol_filter(xx, 21, 3)
                yy = savgol_filter(yy, 21, 3)
                zz = savgol_filter(zz, 21, 3)

            for ax in [x_ax, x_ax_log]:
                if ax:
                    ax.plot(x, xx,
                            alpha=alpha,
                            label=label,
                            ls=ls,
                            color=lc,
                            zorder=1)
            for ax in [y_ax, y_ax_log]:
                if ax:
                    ax.plot(x, yy,
                            alpha=alpha,
                            label=label,
                            ls=ls,
                            color=lc,
                            zorder=1)
            for ax in [z_ax, z_ax_log]:
                if ax:
                    ax.plot(x, zz,
                            alpha=alpha,
                            label=label,
                            ls=ls,
                            color=lc,
                            zorder=1)

    def format_figure(figure, axes, title, files, min_ch, max_ch,
                      ch_step=1, b_field=False, ylabel='', footnote='',
                      x_min=None, x_max=None, y_min=None, y_max=None,
                      incl_legend=True, incl_legend_ls=False, incl_legend_colors=False,
                      style_legend_by='time', color_legend_by='file'):

        for legend in figure.legends:
            legend.remove()

        if not isinstance(files, list):
            files = [files]

        rainbow_colors = cm.jet(np.linspace(0, 1, (int((max_ch - min_ch) / ch_step)) + 1))
        line_styles = ['-', '--', '-.', ':']

        x_ax, x_ax_log = axes.get('X')
        y_ax, y_ax_log = axes.get('Y')
        z_ax, z_ax_log = axes.get('Z')

        # Set the labels
        for ax in [x_ax_log, y_ax_log, z_ax_log]:
            if ax:
                ax.set_xlabel(f"Station")

        if ylabel:
            if x_ax:
                x_ax.set_ylabel(ylabel)
                x_ax_log.set_ylabel(ylabel)
            elif y_ax:
                y_ax.set_ylabel(ylabel)
                y_ax_log.set_ylabel(ylabel)
            else:
                z_ax.set_ylabel(ylabel)
                z_ax_log.set_ylabel(ylabel)
        else:
            if b_field is True:
                if x_ax:
                    x_ax.set_ylabel(f"EM Response\n(nT)")
                    x_ax_log.set_ylabel(f"EM Response\n(nT)")
                elif y_ax:
                    y_ax.set_ylabel(f"EM Response\n(nT)")
                    y_ax_log.set_ylabel(f"EM Response\n(nT)")
                else:
                    z_ax.set_ylabel(f"EM Response\n(nT)")
                    z_ax_log.set_ylabel(f"EM Response\n(nT)")
            else:
                if x_ax:
                    x_ax.set_ylabel(f"EM Response\n(nT/s)")
                    x_ax_log.set_ylabel(f"EM Response\n(nT/s)")
                elif y_ax:
                    y_ax.set_ylabel(f"EM Response\n(nT/s)")
                    y_ax_log.set_ylabel(f"EM Response\n(nT/s)")
                else:
                    z_ax.set_ylabel(f"EM Response\n(nT/s)")
                    z_ax_log.set_ylabel(f"EM Response\n(nT/s)")

        figure.suptitle(title)
        if x_ax and x_ax_log:
            x_ax.set_title(f"X Component")
            x_ax_log.set_title(f"X Component")
        if y_ax and y_ax_log:
            y_ax.set_title(f"Y Component")
            y_ax_log.set_title(f"Y Component")
        if z_ax and z_ax_log:
            z_ax.set_title(f"Z Component")
            z_ax_log.set_title(f"Z Component")

        for ax in figure.axes:
            if x_min is not None and x_max is not None:
                ax.set_xlim([x_min, x_max])
            if y_min is not None and y_max is not None:
                ax.set_ylim([y_min, y_max])

            if ax in [x_ax_log, y_ax_log, z_ax_log]:
                if ax:
                    ymin, ymax = ax.get_ylim()
                    if ymax - ymin < 20:
                        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))

        if incl_legend:
            handles, labels = [], []
            if incl_legend_colors:
                # Use filetype colors in legend, or color by channel (rainbow colors)
                if color_legend_by == 'file':
                    for i, file in enumerate(files):
                        file_type = get_filetype(file)
                        line = Line2D([0], [0], color=colors.get(file_type), linestyle="-")
                        handles.append(line)
                        labels.append(file_type)
                elif color_legend_by == 'time':
                    for i, ch in enumerate(np.arange(min_ch, max_ch + 1, ch_step)):
                        time_label = f"{files[0].ch_times[ch - 1]:.3f}ms"
                        line = Line2D([0], [0], color=rainbow_colors[i], linestyle="-")
                        handles.append(line)
                        labels.append(time_label)
                else:
                    ax_handles, ax_labels = z_ax.get_legend_handles_labels()
                    handles.extend(ax_handles)
                    labels.extend(ax_labels)

            if incl_legend_ls:
                # Add a separator
                if all([incl_legend_ls, incl_legend_colors]):
                    line = Line2D([0], [0], color='w', linestyle='-', alpha=0.0)
                    label = ''
                    handles.append(line)
                    labels.append(label)

                # Use the filetype line styles, or use the channels as different line styles.
                if style_legend_by == 'file':
                    for i, file in enumerate(files):
                        file_type = get_filetype(file)
                        line = Line2D([0], [0], color='k', linestyle=styles.get(file_type))
                        handles.append(line)
                        labels.append(file_type)
                elif style_legend_by == 'time':
                    for i, ch in enumerate(np.arange(min_ch, max_ch + 1, ch_step)):
                        time_label = f"{files[0].ch_times[ch - 1]:.3f}ms"
                        line = Line2D([0], [0], color='k', linestyle=line_styles[(i % len(line_styles))])
                        handles.append(line)
                        labels.append(time_label)
                else:
                    ax_handles, ax_labels = z_ax.get_legend_handles_labels()
                    handles.extend(ax_handles)
                    labels.extend(ax_labels)

            # if not any([incl_legend_ls, incl_legend_colors]):
            #     ax_handles, ax_labels = z_ax.get_legend_handles_labels()
            #     handles.extend(ax_handles)
            #     labels.extend(ax_labels)

            figure.legend(handles, labels, loc='upper right')

            if footnote:
                figure.axes[0].text(0.995, 0.01, footnote,
                                    ha='right',
                                    va='bottom',
                                    size=8,
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
        elif isinstance(file_object, pd.DataFrame):
            return "DataFrame"
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

        base_files = [re.sub(r"\D", "", f) for f in plotting_files if len(f) == 1]
        print(f"Base files found for {plotting_files}: {base_files}")
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
            if ax:
                ax.clear()

    def log_scale(log_axes):
        for ax in log_axes:
            if ax:
                ax.set_yscale('symlog', subs=list(np.arange(2, 10, 1)), linthresh=10, linscale=1. / math.log(10))

    def get_runtime(t):
        runtime = f"{math.floor((time.time() - t) / 60):02.0f}:{(time.time() - t) % 60:02.0f}"
        return runtime

    def get_unique_files(files):
        """
        Return all unique file names.
        :param files: list of lists
        :return: list of str
        """
        unique_filenames = np.unique([f.stem for f in np.concatenate(files)])
        return unique_filenames

    def plot_aspect_ratio():

        def plot_all():
            log_file_path = sample_files.joinpath(r"Aspect Ratio\Aspect ratio log.txt")
            logging_file = open(str(log_file_path), "w+")

            print("Plotting all aspect ratio models")
            logging_file.write("Plotting all aspect ratio models\n")

            maxwell_dir = sample_files.joinpath(r"Aspect Ratio\Maxwell\2m stations")
            mun_dir = sample_files.joinpath(r"Aspect Ratio\MUN")
            plate_dir = sample_files.joinpath(r"Aspect Ratio\PLATE\2m stations")
            irap_dir = sample_files.joinpath(r"Aspect Ratio\IRAP")

            maxwell_files = list(maxwell_dir.glob("*.TEM"))
            mun_files = list(mun_dir.glob("*.DAT"))
            plate_files = list(plate_dir.glob("*.DAT"))
            irap_files = list(irap_dir.glob("*.DAT"))

            base_out_pdf = sample_files.joinpath(r"Aspect ratio")

            unique_files = get_unique_files([maxwell_files, mun_files, plate_files, irap_files])
            small_plate_files = [f for f in unique_files if '150' in f]
            big_plate_files = [f for f in unique_files if '600' in f]

            t = time.time()
            count = 0
            for files, out_pdf in zip([small_plate_files, big_plate_files],
                                      [base_out_pdf.joinpath("Aspect Ratio Models - 150m Plates.PDF"),
                                       base_out_pdf.joinpath("Aspect Ratio Models - 600m Plates.PDF")]):

                with PdfPages(out_pdf) as pdf:

                    for stem in files:
                        print(f"Plotting model {stem} ({count + 1}/{len(unique_files)})")
                        format_files = []

                        max_obj = None
                        mun_obj = None
                        irap_obj = None
                        plate_obj = None

                        max_file = maxwell_dir.joinpath(stem).with_suffix(".TEM")
                        mun_file = mun_dir.joinpath(stem).with_suffix(".DAT")
                        irap_file = irap_dir.joinpath(stem).with_suffix(".DAT")
                        plate_file = plate_dir.joinpath(stem).with_suffix(".DAT")

                        if not max_file.exists():
                            logging_file.write(f"{stem} missing from Maxwell.\n")
                            print(f"{stem} missing from Maxwell.")
                        else:
                            max_obj = TEMFile().parse(max_file)
                            format_files.append(max_obj)

                        if not mun_file.exists():
                            logging_file.write(f"{stem} missing from MUN.\n")
                            print(f"{stem} missing from MUN.")
                        else:
                            mun_obj = MUNFile().parse(mun_file)
                            format_files.append(mun_obj)

                        if not irap_file.exists():
                            logging_file.write(f"{stem} missing from IRAP.\n")
                            print(f"{stem} missing from IRAP.")
                        else:
                            irap_obj = IRAPFile().parse(irap_file)
                            format_files.append(irap_obj)

                        if not plate_file.exists():
                            logging_file.write(f"{stem} missing from PLATE.\n")
                            print(f"{stem} missing from PLATE.")
                        else:
                            plate_obj = PlateFFile().parse(plate_file)
                            format_files.append(plate_obj)

                        if not format_files:
                            logging_file.write(f"No files found for {stem}.\n")
                            print(f"No files found for {stem}.")
                            continue

                        for ch_range in channel_tuples:
                            start_ch, end_ch = ch_range[0], ch_range[1]
                            if ch_range[0] < min_ch:
                                start_ch = min_ch
                            if ch_range[1] > max_ch:
                                end_ch = max_ch
                            print(f"Plotting channel {start_ch} to {end_ch}")

                            if max_obj:
                                plot_obj(ax_dict, max_obj, start_ch, end_ch,
                                         ch_step=channel_step,
                                         station_shift=-400,
                                         data_scaling=1e-6,
                                         lc=colors.get("Maxwell")
                                         )
                            if mun_obj:
                                plot_obj(ax_dict, mun_obj, start_ch, end_ch,
                                         ch_step=channel_step,
                                         station_shift=-200,
                                         alpha=1.,
                                         filter=False,
                                         lc=colors.get("MUN")
                                         )

                            if irap_obj:
                                plot_obj(ax_dict, irap_obj, start_ch, end_ch,
                                         ch_step=channel_step,
                                         alpha=0.6,
                                         lc=colors.get("IRAP")
                                         )

                            if plate_obj:
                                plot_obj(ax_dict, plate_obj, start_ch - 20, end_ch - 20,
                                         ch_step=channel_step,
                                         alpha=0.6,
                                         lc=colors.get("PLATE")
                                         )

                            footnote = "MUN data filtered using Savitzky-Golay filter"
                            format_figure(figure, ax_dict,
                                          f"Aspect Ratio Model\n"
                                          f"{stem}\n"
                                          f"{max_obj.ch_times[start_ch - 1]}ms to {max_obj.ch_times[end_ch - 1]}ms",
                                          format_files, start_ch, end_ch,
                                          x_min=0,
                                          x_max=200,
                                          ch_step=channel_step,
                                          incl_legend=True,
                                          incl_legend_ls=True,
                                          incl_legend_colors=True,
                                          style_legend_by='time',
                                          color_legend_by='file',
                                          footnote="")

                            pdf.savefig(figure, orientation='landscape')
                            clear_axes(axes)
                            log_scale([x_ax_log, y_ax_log, z_ax_log])
                        count += 1

                os.startfile(str(out_pdf))

            print(f"Aspect ratio runtime: {get_runtime(t)}")
            logging_file.write(f"Aspect ratio runtime: {get_runtime(t)}\n")
            logging_file.close()

        def plot_irap_mun():
            log_file_path = sample_files.joinpath(r"Aspect Ratio\Aspect ratio (IRAP vs MUN) log.txt")
            logging_file = open(str(log_file_path), "w+")

            print(f"Plotting IRAP vs MUN")
            logging_file.write(f"Plotting IRAP vs MUN\n")

            mun_dir = sample_files.joinpath(r"Aspect Ratio\MUN")
            irap_dir = sample_files.joinpath(r"Aspect Ratio\IRAP")

            mun_files = list(mun_dir.glob("*.DAT"))
            irap_files = list(irap_dir.glob("*.DAT"))

            base_out_pdf = sample_files.joinpath(r"Aspect ratio")

            unique_files = get_unique_files([mun_files, irap_files])
            small_plate_files = [f for f in unique_files if '150' in f]
            big_plate_files = [f for f in unique_files if '600' in f]

            t = time.time()
            count = 0
            for files, out_pdf in zip([small_plate_files, big_plate_files],
                                      [base_out_pdf.joinpath("Aspect Ratio Models - 150m Plates, IRAP vs MUN.PDF"),
                                       base_out_pdf.joinpath("Aspect Ratio Models - 600m Plates, IRAP vs MUN.PDF")]):

                with PdfPages(out_pdf) as pdf:

                    for stem in files:
                        print(f"Plotting model {stem} ({count + 1}/{len(unique_files)})")
                        format_files = []

                        mun_file = mun_dir.joinpath(stem).with_suffix(".DAT")
                        irap_file = irap_dir.joinpath(stem).with_suffix(".DAT")

                        if not all([mun_file.exists(), irap_file.exists()]):
                            logging_file.write(f"{stem} is not available for both filetypes.\n")
                            print(f"{stem} is not available for both filetypes.")
                            continue
                        else:
                            mun_obj = MUNFile().parse(mun_file)
                            irap_obj = IRAPFile().parse(irap_file)
                            format_files.append(mun_obj)
                            format_files.append(irap_obj)

                        if not format_files:
                            logging_file.write(f"No files found for {stem}.\n")
                            print(f"No files found for {stem}.")
                            continue

                        for ch_range in channel_tuples:
                            start_ch, end_ch = ch_range[0], ch_range[1]
                            if ch_range[0] < min_ch:
                                start_ch = min_ch
                            if ch_range[1] > max_ch:
                                end_ch = max_ch
                            print(f"Plotting channel {start_ch} to {end_ch}")

                            if mun_obj:
                                plot_obj(ax_dict, mun_obj, start_ch, end_ch,
                                         ch_step=channel_step,
                                         station_shift=-200,
                                         alpha=1.,
                                         filter=False,
                                         lc=colors.get("MUN")
                                         )

                            if irap_obj:
                                plot_obj(ax_dict, irap_obj, start_ch, end_ch,
                                         ch_step=channel_step,
                                         alpha=0.6,
                                         lc=colors.get("IRAP")
                                         )

                            footnote = ""  # f"MUN data filtered using Savitzky-Golay filter"
                            format_figure(figure, ax_dict,
                                          f"Aspect Ratio Model\n"
                                          f"{stem}\n"
                                          f"{mun_obj.ch_times[start_ch - 1]}ms to {mun_obj.ch_times[end_ch - 1]}ms",
                                          format_files, start_ch, end_ch,
                                          x_min=0,
                                          x_max=200,
                                          ch_step=channel_step,
                                          incl_legend=True,
                                          incl_legend_ls=True,
                                          incl_legend_colors=True,
                                          style_legend_by='time',
                                          color_legend_by='file',
                                          footnote=footnote)

                            pdf.savefig(figure, orientation='landscape')
                            clear_axes(axes)
                            log_scale([x_ax_log, y_ax_log, z_ax_log])
                        count += 1

                os.startfile(str(out_pdf))

            print(f"Aspect ratio runtime: {get_runtime(t)}")
            logging_file.write(f"Aspect ratio (IRAP vs MUN) runtime: {get_runtime(t)}\n")
            logging_file.close()

        def plot_100m_below_surface():
            log_file_path = sample_files.joinpath(r"Aspect Ratio\Aspect ratio 100m below surface log.txt")
            logging_file = open(str(log_file_path), "w+")

            print("Plotting 100m below surface")
            logging_file.write("Plotting 100m below surface\n")

            maxwell_dir = sample_files.joinpath(
                r"Aspect ratio\Maxwell\Aspect Ratio Plates - 100m below surface")
            plate_dir = sample_files.joinpath(
                r"Aspect ratio\PLATE\two plates from aspect ratio test moved 100m below surface")

            maxwell_files = list(maxwell_dir.glob("*.TEM"))
            plate_files = list(plate_dir.glob("*.DAT"))
            unique_files = get_unique_files([maxwell_files, plate_files])

            out_pdf = sample_files.joinpath(r"Aspect Ratio\Aspect Ratio Models - 100m Below Surface.PDF")
            t = time.time()
            count = 0
            with PdfPages(out_pdf) as pdf:

                for stem in unique_files:
                    print(f"Plotting model {stem} ({count + 1}/{len(unique_files)})")
                    format_files = []

                    max_file = maxwell_dir.joinpath(stem).with_suffix(".TEM")
                    plate_file = plate_dir.joinpath(stem).with_suffix(".DAT")

                    if not all([max_file.exists(), plate_file.exists()]):
                        logging_file.write(f"{stem} is not available for both filetypes.\n")
                        print(f"{stem} is not available for both filetypes.")
                        continue
                    else:
                        max_obj = TEMFile().parse(max_file)
                        plate_obj = PlateFFile().parse(plate_file)
                        format_files.append(max_obj)
                        format_files.append(plate_obj)

                    if not format_files:
                        logging_file.write(f"No files found for {stem}.\n")
                        print(f"No files found for {stem}.")
                        continue

                    for ch_range in channel_tuples:
                        start_ch, end_ch = ch_range[0], ch_range[1]
                        if ch_range[0] < min_ch:
                            start_ch = min_ch
                        if ch_range[1] > max_ch:
                            end_ch = max_ch
                        print(f"Plotting channel {start_ch} to {end_ch}")

                        if max_obj:
                            plot_obj(ax_dict, max_obj, start_ch, end_ch,
                                     ch_step=channel_step,
                                     station_shift=-400,
                                     data_scaling=1e-6,
                                     lc=colors.get("Maxwell")
                                     )

                        if plate_obj:
                            plot_obj(ax_dict, plate_obj, start_ch - 20, end_ch - 20,
                                     ch_step=channel_step,
                                     alpha=0.6,
                                     lc=colors.get("PLATE")
                                     )

                        format_figure(figure, ax_dict,
                                      f"Aspect Ratio Model: 100m Below Surface\n"
                                      f"{stem}\n"
                                      f"{max_obj.ch_times[start_ch - 1]}ms to {max_obj.ch_times[end_ch - 1]}ms",
                                      format_files, start_ch, end_ch,
                                      x_min=None,
                                      x_max=None,
                                      ch_step=channel_step,
                                      incl_legend=True,
                                      incl_legend_ls=True,
                                      incl_legend_colors=True,
                                      style_legend_by='time',
                                      color_legend_by='file',
                                      footnote="")

                        pdf.savefig(figure, orientation='landscape')
                        clear_axes(axes)
                        log_scale([x_ax_log, y_ax_log, z_ax_log])
                    count += 1

            os.startfile(str(out_pdf))

            print(f"Aspect ratio runtime: {get_runtime(t)}")
            logging_file.write(f"Aspect ratio 100m below surface runtime: {get_runtime(t)}\n")
            logging_file.close()

        def plot_horizontal():
            log_file_path = sample_files.joinpath(r"Aspect Ratio\Aspect ratio horizontal plates log.txt")
            logging_file = open(str(log_file_path), "w+")

            print("Plotting horizontal plates")
            logging_file.write("Plotting horizontal plates\n")

            figure, ((x_ax,  z_ax), (x_ax_log, z_ax_log)) = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='none')
            ax_dict = {"X": (x_ax, x_ax_log), "Y": (None, None), "Z": (z_ax, z_ax_log)}
            axes = [x_ax, z_ax, x_ax_log, z_ax_log]
            figure.set_size_inches((11 * 1.33 * 1.33, 8.5 * 1.33))
            log_scale([x_ax_log, z_ax_log])

            maxwell_dir = sample_files.joinpath(
                r"Aspect ratio\Maxwell\Horizontal Plates - 100m below surface")
            plate_dir = sample_files.joinpath(
                r"Aspect ratio\PLATE\two horizontal plates centred 100m below 400m x 400m loop")

            maxwell_files = list(maxwell_dir.glob("*.TEM"))
            plate_files = list(plate_dir.glob("*.DAT"))
            unique_files = get_unique_files([maxwell_files, plate_files])

            out_pdf = sample_files.joinpath(r"Aspect Ratio\Aspect Ratio Models - Horizontal Plates.PDF")
            t = time.time()
            count = 0
            with PdfPages(out_pdf) as pdf:

                for stem in unique_files:
                    print(f"Plotting model {stem} ({count + 1}/{len(unique_files)})")
                    format_files = []

                    max_file = maxwell_dir.joinpath(stem).with_suffix(".TEM")
                    plate_file = plate_dir.joinpath(stem).with_suffix(".DAT")

                    if not all([max_file.exists(), plate_file.exists()]):
                        logging_file.write(f"{stem} is not available for both filetypes.\n")
                        print(f"{stem} is not available for both filetypes.")
                        continue
                    else:
                        max_obj = TEMFile().parse(max_file)
                        plate_obj = PlateFFile().parse(plate_file)
                        format_files.append(max_obj)
                        format_files.append(plate_obj)

                    if not format_files:
                        logging_file.write(f"No files found for {stem}.\n")
                        print(f"No files found for {stem}.")
                        continue

                    for ch_range in channel_tuples:
                        start_ch, end_ch = ch_range[0], ch_range[1]
                        if ch_range[0] < min_ch:
                            start_ch = min_ch
                        if ch_range[1] > max_ch:
                            end_ch = max_ch
                        print(f"Plotting channel {start_ch} to {end_ch}")

                        if max_obj:
                            plot_obj(ax_dict, max_obj, start_ch, end_ch,
                                     ch_step=channel_step,
                                     station_shift=0,
                                     data_scaling=1e-6,
                                     lc=colors.get("Maxwell")
                                     )

                        if plate_obj:
                            plot_obj(ax_dict, plate_obj, start_ch - 20, end_ch - 20,
                                     ch_step=channel_step,
                                     alpha=0.6,
                                     lc=colors.get("PLATE")
                                     )

                        format_figure(figure, ax_dict,
                                      f"Aspect Ratio Model: Horizontal Plates\n"
                                      f"{stem}\n"
                                      f"{max_obj.ch_times[start_ch - 1]}ms to {max_obj.ch_times[end_ch - 1]}ms",
                                      format_files, start_ch, end_ch,
                                      x_min=None,
                                      x_max=None,
                                      ch_step=channel_step,
                                      incl_legend=True,
                                      incl_legend_ls=True,
                                      incl_legend_colors=True,
                                      style_legend_by='time',
                                      color_legend_by='file',
                                      footnote="")

                        pdf.savefig(figure, orientation='landscape')
                        clear_axes(axes)
                        log_scale([x_ax_log, y_ax_log, z_ax_log])
                    count += 1

            os.startfile(str(out_pdf))

            print(f"Aspect ratio runtime: {get_runtime(t)}")
            logging_file.write(f"Aspect ratio runtime: {get_runtime(t)}\n")
            logging_file.close()

        figure, ((x_ax, y_ax, z_ax), (x_ax_log, y_ax_log, z_ax_log)) = plt.subplots(nrows=2, ncols=3, sharex='all', sharey='none')
        ax_dict = {"X": (x_ax, x_ax_log), "Y": (y_ax, y_ax_log), "Z": (z_ax, z_ax_log)}
        axes = [x_ax, y_ax, z_ax, x_ax_log, y_ax_log, z_ax_log]
        figure.set_size_inches((11 * 1.33 * 1.33, 8.5 * 1.33))
        log_scale([x_ax_log, y_ax_log, z_ax_log])

        # global min_ch, max_ch, channel_step
        min_ch, max_ch = 21, 44
        channel_step = 1
        num_chs = 4
        channel_tuples = list(zip(np.arange(min_ch, max_ch, num_chs - 1),
                                  np.arange(min_ch + num_chs - 1, max_ch + num_chs - 1, num_chs - 1)))

        plot_all()
        plot_irap_mun()
        # plot_100m_below_surface()
        # plot_horizontal()

    def plot_two_way_induction():

        def plot_all(conductance, start_file=False):
            log_file_path = sample_files.joinpath(r"Two-way induction\300x100\100S\Two-way induction log.txt")
            logging_file = open(str(log_file_path), "w+")

            print(f"Plotting all {conductance} two-way induction models")
            logging_file.write(f">>Plotting all {conductance} two-way induction models\n\n")
            figure, ((x_ax, z_ax), (x_ax_log, z_ax_log)) = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='none')
            ax_dict = {"X": (x_ax, x_ax_log), "Y": (None, None), "Z": (z_ax, z_ax_log)}
            axes = [x_ax, z_ax, x_ax_log, z_ax_log]
            figure.set_size_inches((11 * 1.33 * 1.33, 8.5 * 1.33))
            log_scale([x_ax_log, z_ax_log])

            maxwell_dir = sample_files.joinpath(fr"Two-way induction\300x100\{conductance}\Maxwell")
            mun_dir = sample_files.joinpath(fr"Two-way induction\300x100\{conductance}\MUN")
            plate_dir = sample_files.joinpath(fr"Two-way induction\300x100\{conductance}\PLATE")

            maxwell_files = list(maxwell_dir.glob("*.TEM"))
            mun_files = list(mun_dir.glob("*.DAT"))
            plate_files = list(plate_dir.glob("*.DAT"))

            out_pdf = sample_files.joinpath(fr"Two-way induction\Two-Way Induction - 300x100m, {conductance}.PDF")

            unique_files = os_sorted(get_unique_files([maxwell_files, mun_files, plate_files]))

            count = 0
            with PdfPages(out_pdf) as pdf:
                for stem in unique_files:
                    print(f"Plotting model {stem} ({count + 1}/{len(unique_files)})")
                    format_files = []

                    max_obj = None
                    mun_obj = None
                    plate_obj = None

                    max_file = maxwell_dir.joinpath(stem).with_suffix(".TEM")
                    mun_file = mun_dir.joinpath(stem).with_suffix(".DAT")
                    plate_file = plate_dir.joinpath(stem).with_suffix(".DAT")

                    if not max_file.exists():
                        logging_file.write(f"{stem} missing from Maxwell.\n")
                        print(f"{stem} missing from Maxwell.")
                    else:
                        max_obj = TEMFile().parse(max_file)
                        format_files.append(max_obj)

                    if not mun_file.exists():
                        logging_file.write(f"{stem} missing from MUN.\n")
                        print(f"{stem} missing from MUN.")
                    else:
                        mun_obj = MUNFile().parse(mun_file)
                        format_files.append(mun_obj)

                    if not plate_file.exists():
                        logging_file.write(f"{stem} missing from PLATE.\n")
                        print(f"{stem} missing from PLATE.")
                    else:
                        plate_obj = PlateFFile().parse(plate_file)
                        format_files.append(plate_obj)

                    if not format_files:
                        logging_file.write(f"No files found for {stem}.")
                        print(f"No files found for {stem}.")
                        continue

                    for ch_range in channel_tuples:
                        start_ch, end_ch = ch_range[0], ch_range[1]
                        if ch_range[0] < min_ch:
                            start_ch = min_ch
                        if ch_range[1] > max_ch:
                            end_ch = max_ch
                        print(f"Plotting channel {start_ch} to {end_ch}")

                        if max_obj:
                            plot_obj(ax_dict, max_obj, start_ch, end_ch,
                                     ch_step=channel_step,
                                     station_shift=0,
                                     data_scaling=1e-6,
                                     lc=colors.get("Maxwell")
                                     )
                        if mun_obj:
                            plot_obj(ax_dict, mun_obj, start_ch, end_ch,
                                     ch_step=channel_step,
                                     station_shift=300,
                                     alpha=1.,
                                     filter=False,
                                     lc=colors.get("MUN")
                                     )

                        if plate_obj:
                            plot_obj(ax_dict, plate_obj, start_ch - 20, end_ch - 20,
                                     ch_step=channel_step,
                                     alpha=0.6,
                                     lc=colors.get("PLATE")
                                     )

                        format_figure(figure, ax_dict,
                                      f"Two-Way Induction\n"
                                      f"300x100m, {conductance}\n"
                                      f"{stem}\n"
                                      f"{max_obj.ch_times[start_ch - 1]}ms to {max_obj.ch_times[end_ch - 1]}ms",
                                      format_files, start_ch, end_ch,
                                      x_min=None,
                                      x_max=None,
                                      ch_step=channel_step,
                                      incl_legend=True,
                                      incl_legend_ls=True,
                                      incl_legend_colors=True,
                                      style_legend_by='time',
                                      color_legend_by='file',
                                      footnote="")

                        pdf.savefig(figure, orientation='landscape')
                        clear_axes(axes)
                        log_scale([x_ax_log, z_ax_log])
                    count += 1

                if start_file:
                    os.startfile(str(out_pdf))

                runtime = get_runtime(t)
                print(f"Two-way induction {conductance} runtime: {runtime}")
                logging_file.write(f"Two-way induction {conductance} runtime: {runtime}\n")
                logging_file.close()

        min_ch, max_ch = 21, 44
        channel_step = 1
        num_chs = 4
        channel_tuples = list(zip(np.arange(min_ch, max_ch, num_chs - 1),
                                  np.arange(min_ch + num_chs - 1, max_ch + num_chs - 1, num_chs - 1)))

        t = time.time()

        plot_all('100S', start_file=False)
        plot_all('1000S', start_file=True)

    def plot_run_on_effect():

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

        plot_run_on_comparison()
        plot_run_on_convergence()
        tabulate_run_on_convergence()

    def plot_infinite_thin_sheet():

        def compare_maxwell_ribbons(start_file=False):
            log_file_path = sample_files.joinpath(
                r"Infinite thin sheet\Infinite Thin Sheet - Maxwell Ribbon Comparison log.txt")
            logging_file = open(str(log_file_path), "w+")

            print(f"Comparing Maxwell ribbons for infinite thin sheet models")
            logging_file.write(f">>Comparing Maxwell ribbons for infinite thin sheet models\n\n")

            figure, ((x_ax, y_ax, z_ax), (x_ax_log, y_ax_log, z_ax_log)) = plt.subplots(nrows=2, ncols=3, sharex='all', sharey='none')
            ax_dict = {"X": (x_ax, x_ax_log), "Y": (y_ax, y_ax_log), "Z": (z_ax, z_ax_log)}
            axes = [x_ax, y_ax, z_ax, x_ax_log, y_ax_log, z_ax_log]
            figure.set_size_inches((11 * 1.33 * 1.33, 8.5 * 1.33))
            log_scale([x_ax_log, y_ax_log, z_ax_log])

            t = time.time()

            out_pdf = sample_files.joinpath(r"Infinite thin Sheet\Infinite Thin Sheet - Maxwell Ribbon Comparison.PDF")

            folder_10 =sample_files.joinpath(r"Infinite thin Sheet\Maxwell\10 Ribbons")
            folder_50 =sample_files.joinpath(r"Infinite thin Sheet\Maxwell\50 Ribbons")

            files_10 = os_sorted(list(folder_10.glob("*.tem")))
            files_50 = os_sorted(list(folder_50.glob("*.tem")))

            min_ch, max_ch = 21, 44
            channel_step = 1
            num_chs = 4
            channel_tuples = list(zip(np.arange(min_ch, max_ch, num_chs - 1),
                                      np.arange(min_ch + num_chs - 1, max_ch + num_chs - 1, num_chs - 1)))

            count = 0
            with PdfPages(out_pdf) as pdf:
                format_files = []
                for filepath_10, filepath_50 in list(zip(files_10, files_50))[:2]:
                    print(f"Plotting set {count + 1}/{len(files_10)}")
                    obj_10 = TEMFile().parse(filepath_10)
                    obj_50 = TEMFile().parse(filepath_50)
                    format_files.extend([obj_10, obj_50])

                    for ch_range in channel_tuples:
                        start_ch, end_ch = ch_range[0], ch_range[1]
                        if ch_range[0] < min_ch:
                            start_ch = min_ch
                        if ch_range[1] > max_ch:
                            end_ch = max_ch
                        print(f"Plotting channel {start_ch} to {end_ch}")

                        plot_obj(ax_dict, obj_10, start_ch, end_ch,
                                 name="10 Ribbons",
                                 ch_step=channel_step,
                                 station_shift=0,
                                 data_scaling=1e-6,
                                 lc="b"
                                 )

                        plot_obj(ax_dict, obj_50, start_ch, end_ch,
                                 name="50 Ribbons",
                                 ch_step=channel_step,
                                 station_shift=0,
                                 data_scaling=1e-6,
                                 lc="r",
                                 alpha=0.6
                                 )

                        format_figure(figure, ax_dict,
                                      f"Infinite Thin Sheet\n"
                                      f"Ribbon Comparison\n"
                                      f"{obj_10.ch_times[start_ch - 1]}ms to {obj_10.ch_times[end_ch - 1]}ms",
                                      format_files, start_ch, end_ch,
                                      x_min=None,
                                      x_max=None,
                                      ch_step=channel_step,
                                      incl_legend=True,
                                      incl_legend_ls=True,
                                      incl_legend_colors=True,
                                      style_legend_by='time',
                                      color_legend_by='line',
                                      footnote="")

                        pdf.savefig(figure, orientation='landscape')
                        clear_axes(axes)
                        log_scale([x_ax_log, y_ax_log, z_ax_log])
                    count += 1

            if start_file:
                os.startfile(str(out_pdf))

            runtime = get_runtime(t)
            print(f"Maxwell infinite thin sheet ribbon comparison runtime: {runtime}")
            logging_file.write(f"Maxwell infinite thin sheet ribbon comparison runtime: {runtime}\n")
            logging_file.close()

        def compare_step_on_with_theory(filetype, start_file=False):

            def spline_data(obj, new_ch_times):
                old_ch_times = obj.ch_times / 1000
                new_ch_times = np.arange(old_ch_times.min(), old_ch_times.max(), 1e-3)
                channels = [f"CH{num}" for num in range(1, len(old_ch_times) + 1)]
                spline_function = interpolate.splrep(old_ch_times, obj.data.loc[1, channels], s=0)
                splined_data = interpolate.splev(new_ch_times, spline_function, der=1)
                fig, ax = plt.subplots()
                ax.plot(old_ch_times, obj.data.loc[1, channels], "ro",
                        markerSize=5,
                        label="MUN data")
                ax.plot(new_ch_times, splined_data, "b+-",
                        alpha=0.5,
                        markerSize=5,
                        label="Spline",
                        zorder=-1)
                # ax.set_xlim(new_ch_times.min(), new_ch_times.max())
                ax.legend()
                plt.show()
                print()

            def plot_theory(x_df, z_df, start_ch, end_ch):
                x = x_df.Position

                for ind, (_, ch_response) in enumerate(x_df.iloc[:, start_ch + 1: end_ch + 1].iteritems()):
                    if ind == 0:
                        label = f"Theory"
                    else:
                        label = None

                    theory_x = ch_response.values

                    for ax in [x_ax, x_ax_log]:
                        ax.plot(x, theory_x,
                                color="r",
                                alpha=0.6,
                                label=label,
                                zorder=1)

                for ind, (_, ch_response) in enumerate(z_df.iloc[:, start_ch + 1: end_ch + 1].iteritems()):
                    if ind == 0:
                        label = f"Theory"
                    else:
                        label = None

                    theory_z = ch_response.values

                    for ax in [z_ax, z_ax_log]:
                        ax.plot(x, theory_z,
                                color="r",
                                alpha=0.6,
                                label=label,
                                zorder=1)

            log_file_path = sample_files.joinpath(
                r"Infinite thin sheet\Infinite Thin Sheet - {filetype} vs Theory log.txt")
            logging_file = open(str(log_file_path), "w+")

            print(f"Comparing {filetype} with theory for infinite thin sheet models")
            logging_file.write(f">>Comparing {filetype} with theory for infinite thin sheet models\n\n")

            figure, ((x_ax, z_ax), (x_ax_log, z_ax_log)) = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='none')
            ax_dict = {"X": (x_ax, x_ax_log), "Y": (None, None), "Z": (z_ax, z_ax_log)}
            axes = [x_ax, z_ax, x_ax_log, z_ax_log]
            figure.set_size_inches((11 * 1.33 * 1.33, 8.5 * 1.33))
            log_scale([x_ax_log, z_ax_log])

            t = time.time()

            conductances = ["1S", "10S", "100S"]
            measurements = ["dBdt", "B"]

            min_ch, max_ch = 1, 32
            channel_step = 1
            num_chs = 4
            channel_tuples = list(zip(np.arange(min_ch, max_ch, num_chs - 1),
                                      np.arange(min_ch + num_chs - 1, max_ch + num_chs - 1, num_chs - 1)))

            for measurement in measurements:
                for conductance in conductances:
                    print(f"Plotting {measurement} at {conductance}.")
                    out_pdf = sample_files.joinpath(
                        fr"Infinite thin Sheet\Infinite Thin Sheet - {filetype} vs Theory ({measurement}, {conductance}).PDF")

                    file_dir = sample_files.joinpath(fr"Infinite thin sheet\{filetype}\{measurement}")
                    theory_dir = sample_files.joinpath(fr"Infinite thin sheet\Theory\{measurement}")

                    theory_x_file = theory_dir.joinpath(fr"Infinite sheet {conductance} {measurement} X.xlsx")
                    theory_z_file = theory_dir.joinpath(fr"Infinite sheet {conductance} {measurement} Z.xlsx")
                    assert all([file_dir.exists(), theory_dir.exists(), theory_x_file.exists(),
                                theory_z_file.exists()]), f"One or more files or directories don't exist."

                    x_df = pd.read_excel(theory_x_file, header=4, engine='openpyxl').dropna(axis=1)
                    z_df = pd.read_excel(theory_z_file, header=4, engine='openpyxl').dropna(axis=1)

                    files = os_sorted(list(file_dir.glob(f"*{conductance}{extensions.get(filetype)}")))
                    if not files:
                        raise ValueError(f"No files found for {filetype} {conductance} {measurement}.")

                    print(f"{len(files)} files found.")
                    count = 0
                    with PdfPages(out_pdf) as pdf:
                        format_files = []
                        for file in files:
                            print(f"Plotting {file.stem} ({count +1}/{len(files)})")

                            print(f"Comparing {filetype} file {'/'.join(file.parts[-2:])} with theory files {theory_x_file.stem}, {theory_z_file.stem}")
                            if filetype == "Maxwell":
                                obj = TEMFile().parse(file)
                                xmin, xmax = obj.data.STATION.min(), obj.data.STATION.max()
                            else:
                                obj = MUNFile().parse(file)
                                obj.data = spline_data(obj, x_df.columns[1:].astype(float))
                                xmin, xmax = obj.data.Station.astype(float).min(), obj.data.Station.astype(float).max()
                                print(xmin, xmax)
                            format_files.append(obj)

                            for ch_range in channel_tuples:
                                start_ch, end_ch = ch_range[0], ch_range[1]
                                if ch_range[0] < min_ch:
                                    start_ch = min_ch
                                if ch_range[1] > max_ch:
                                    end_ch = max_ch
                                print(f"Plotting channel {start_ch} to {end_ch}")

                                plot_obj(ax_dict, obj, start_ch + 20, end_ch + 20,
                                         name=filetype,
                                         ch_step=channel_step,
                                         station_shift=0,
                                         data_scaling=-1 if measurement == 'B' and filetype == "Maxwell" else 1,
                                         lc=colors.get(filetype),
                                         alpha=1.
                                         )

                                plot_theory(x_df, z_df, start_ch, end_ch)

                                size = re.search(r"(\d+x\d+).*", file.stem).group(1)
                                footnote = ""
                                if measurement == "B":
                                    footnote = f"{filetype} file data multiplied by -1."

                                format_figure(figure, ax_dict,
                                              f"Infinite Thin Sheet: Current Step-On, {filetype} vs Theory\n"
                                              f"{size} {measurement}, {conductance}\n"
                                              f"{obj.ch_times[start_ch - 1]}ms to {obj.ch_times[end_ch - 1]}ms",
                                              format_files, start_ch, end_ch,
                                              x_min=xmin,
                                              x_max=xmax,
                                              b_field=True if measurement == 'B' else False,
                                              ch_step=channel_step,
                                              incl_legend=True,
                                              incl_legend_ls=True,
                                              incl_legend_colors=True,
                                              style_legend_by='time',
                                              color_legend_by='line',
                                              footnote=footnote)

                                pdf.savefig(figure, orientation='landscape')
                                clear_axes(axes)
                                log_scale([x_ax_log, z_ax_log])
                            count += 1

                    if start_file:
                        os.startfile(str(out_pdf))

            runtime = get_runtime(t)
            print(f"{filetype} infinite thin sheet theory comparison runtime: {runtime}")
            logging_file.write(f"{filetype} infinite thin sheet theory comparison runtime: {runtime}\n")
            logging_file.close()

        # compare_maxwell_ribbons(start_file=True)
        # compare_step_on_with_theory("Maxwell", start_file=True)
        compare_step_on_with_theory("MUN", start_file=True)

    def plot_infinite_half_sheet():

        def plot_loop(title, start_file=False):
            log_file_path = sample_files.joinpath(fr"Infinite half sheet\Infinite half sheet ({title}).log")
            logging_file = open(str(log_file_path), "w+")

            print(f"Plotting infinite half sheet ({title})")
            logging_file.write(f">>Plotting infinite half sheet ({title}))\n\n")
            figure, ((x_ax, y_ax, z_ax), (x_ax_log, y_ax_log, z_ax_log)) = plt.subplots(nrows=2, ncols=3, sharex='all', sharey='none')
            ax_dict = {"X": (x_ax, x_ax_log), "Y": (y_ax, y_ax_log), "Z": (z_ax, z_ax_log)}
            axes = [x_ax, y_ax, z_ax, x_ax_log, y_ax_log, z_ax_log]
            figure.set_size_inches((11 * 1.33 * 1.33, 8.5 * 1.33))
            log_scale([x_ax_log, y_ax_log, z_ax_log])

            max_dir = sample_files.joinpath(fr"Infinite half sheet\Maxwell\{title}")
            mun_dir = sample_files.joinpath(fr"Infinite half sheet\MUN\{title}")

            max_files = list(max_dir.glob("*.TEM"))
            mun_files = list(mun_dir.glob("*.DAT"))

            out_pdf = sample_files.joinpath(fr"Infinite half sheet\Infinite half sheet ({title}).PDF")
            # out_pdf = sample_files.joinpath(fr"Infinite half sheet\{title} - Savitzky-Golay Filter.PDF")

            unique_files = os_sorted(get_unique_files([max_files, mun_files]))

            count = 0
            with PdfPages(out_pdf) as pdf:
                for stem in unique_files:
                    print(f"Plotting model {stem} ({count + 1}/{len(unique_files)})")
                    format_files = []

                    max_obj = None
                    mun_obj = None

                    max_file = max_dir.joinpath(stem).with_suffix(".TEM")
                    mun_file = mun_dir.joinpath(stem).with_suffix(".DAT")

                    if not max_file.exists():
                        logging_file.write(f"{stem} missing from Maxwell.\n")
                        print(f"{stem} missing from Maxwell.")
                    else:
                        max_obj = TEMFile().parse(max_file)
                        format_files.append(max_obj)

                    if not max_file.exists():
                        logging_file.write(f"{stem} missing from MUN.\n")
                        print(f"{stem} missing from MUN.")
                    else:
                        mun_obj = MUNFile().parse(mun_file)
                        format_files.append(mun_obj)

                    if not format_files:
                        logging_file.write(f"No files found for {stem}.")
                        print(f"No files found for {stem}.")
                        continue

                    for ch_range in channel_tuples:
                        start_ch, end_ch = ch_range[0], ch_range[1]
                        if ch_range[0] < min_ch:
                            start_ch = min_ch
                        if ch_range[1] > max_ch:
                            end_ch = max_ch
                        print(f"Plotting channel {start_ch} to {end_ch}")

                        filter = True
                        if filter is True:
                            footnote = "MUN data filtered using Savitzky-Golay filter"
                        else:
                            footnote = ""

                        if max_obj:
                            plot_obj(ax_dict, max_obj, start_ch, end_ch,
                                     ch_step=channel_step,
                                     station_shift=-200,
                                     data_scaling=1e-6,
                                     lc=colors.get("Maxwell")
                                     )

                        if mun_obj:
                            plot_obj(ax_dict, mun_obj, start_ch, end_ch,
                                     ch_step=channel_step,
                                     station_shift=0,
                                     # data_scaling=1e-6,
                                     lc=colors.get("MUN"),
                                     filter=filter,
                                     )

                        format_figure(figure, ax_dict,
                                      f"Infinite Half Sheet: {title}\n"
                                      f"{stem}\n"
                                      f"{format_files[0].ch_times[start_ch - 1]}ms to {format_files[0].ch_times[end_ch - 1]}ms",
                                      format_files, start_ch, end_ch,
                                      x_min=None,
                                      x_max=None,
                                      ch_step=channel_step,
                                      incl_legend=True,
                                      incl_legend_ls=True,
                                      incl_legend_colors=True,
                                      style_legend_by='time',
                                      color_legend_by='file',
                                      footnote=footnote)

                        # """Comparing the filter"""
                        # plot_obj(ax_dict, mun_obj, start_ch, end_ch,
                        #          ch_step=channel_step,
                        #          station_shift=0,
                        #          filter=True,
                        #          name="Filtered",
                        #          alpha=1.,
                        #          ls="-"
                        #          )
                        #
                        # plot_obj(ax_dict, mun_obj, start_ch, end_ch,
                        #          ch_step=channel_step,
                        #          station_shift=0,
                        #          filter=False,
                        #          name="Original",
                        #          alpha=0.5,
                        #          ls=":"
                        #          )
                        #
                        # format_figure(figure, ax_dict,
                        #               f"{title} - Savitzki-Golay Filter\n"
                        #               f"{stem}\n"
                        #               f"{mun_obj.ch_times[start_ch - 1]}ms to {mun_obj.ch_times[end_ch - 1]}ms",
                        #               format_files, start_ch, end_ch,
                        #               x_min=None,
                        #               x_max=None,
                        #               ch_step=channel_step,
                        #               incl_legend=True,
                        #               incl_legend_ls=True,
                        #               incl_legend_colors=True,
                        #               style_legend_by='line',
                        #               color_legend_by='time',
                        #               footnote="")

                        pdf.savefig(figure, orientation='landscape')
                        clear_axes(axes)
                        log_scale([x_ax_log, y_ax_log, z_ax_log])
                    count += 1

                if start_file:
                    os.startfile(str(out_pdf))

                runtime = get_runtime(t)
                print(f"Infinite half sheet ({title}) runtime: {runtime}")
                logging_file.write(f"Infinite half sheet ({title}) runtime: {runtime}\n")
                logging_file.close()

        # def plot_loop_on_175w(start_file=False):
        #     log_file_path = sample_files.joinpath(r"Infinite half sheet\Infinite half sheet (loop on 175W) log.txt")
        #     logging_file = open(str(log_file_path), "w+")
        #
        #     print(f"Plotting infinite half sheet with loop on 175W")
        #     logging_file.write(f">>Plotting infinite half sheet with loop on 175W\n\n")
        #     figure, ((x_ax, y_ax, z_ax), (x_ax_log, y_ax_log, z_ax_log)) = plt.subplots(nrows=2, ncols=3, sharex='all', sharey='none')
        #     ax_dict = {"X": (x_ax, x_ax_log), "Y": (y_ax, y_ax_log), "Z": (z_ax, z_ax_log)}
        #     axes = [x_ax, y_ax, z_ax, x_ax_log, y_ax_log, z_ax_log]
        #     figure.set_size_inches((11 * 1.33 * 1.33, 8.5 * 1.33))
        #     log_scale([x_ax_log, y_ax_log, z_ax_log])
        #
        #     maxwell_dir = sample_files.joinpath(r"Infinite half sheet\Loop Centered at 175W")
        #
        #     maxwell_files = list(maxwell_dir.glob("*.TEM"))
        #
        #     out_pdf = sample_files.joinpath(r"Infinite half sheet\Infinite half sheet (loop on 175W).PDF")
        #
        #     unique_files = os_sorted(get_unique_files([maxwell_files]))
        #
        #     count = 0
        #     with PdfPages(out_pdf) as pdf:
        #         for stem in unique_files:
        #             print(f"Plotting model {stem} ({count + 1}/{len(unique_files)})")
        #             format_files = []
        #
        #             max_obj = None
        #
        #             max_file = maxwell_dir.joinpath(stem).with_suffix(".TEM")
        #
        #             if not max_file.exists():
        #                 logging_file.write(f"{stem} missing from Maxwell.\n")
        #                 print(f"{stem} missing from Maxwell.")
        #             else:
        #                 max_obj = TEMFile().parse(max_file)
        #                 format_files.append(max_obj)
        #
        #             if not format_files:
        #                 logging_file.write(f"No files found for {stem}.")
        #                 print(f"No files found for {stem}.")
        #                 continue
        #
        #             for ch_range in channel_tuples:
        #                 start_ch, end_ch = ch_range[0], ch_range[1]
        #                 if ch_range[0] < min_ch:
        #                     start_ch = min_ch
        #                 if ch_range[1] > max_ch:
        #                     end_ch = max_ch
        #                 print(f"Plotting channel {start_ch} to {end_ch}")
        #
        #                 if max_obj:
        #                     plot_obj(ax_dict, max_obj, start_ch, end_ch,
        #                              ch_step=channel_step,
        #                              station_shift=0,
        #                              data_scaling=1e-6,
        #                              lc=colors.get("Maxwell")
        #                              )
        #
        #                 format_figure(figure, ax_dict,
        #                               f"Infinite Half Sheet: Loop On 175W\n"
        #                               f"{stem}\n"
        #                               f"{max_obj.ch_times[start_ch - 1]}ms to {max_obj.ch_times[end_ch - 1]}ms",
        #                               format_files, start_ch, end_ch,
        #                               x_min=None,
        #                               x_max=None,
        #                               ch_step=channel_step,
        #                               incl_legend=True,
        #                               incl_legend_ls=True,
        #                               incl_legend_colors=True,
        #                               style_legend_by='time',
        #                               color_legend_by='file',
        #                               footnote="")
        #
        #                 pdf.savefig(figure, orientation='landscape')
        #                 clear_axes(axes)
        #                 log_scale([x_ax_log, y_ax_log, z_ax_log])
        #             count += 1
        #
        #         if start_file:
        #             os.startfile(str(out_pdf))
        #
        #         runtime = get_runtime(t)
        #         print(f"Infinite half sheet (loop on 175W) runtime: {runtime}")
        #         logging_file.write(f"Infinite half sheet (loop on 175W) runtime: {runtime}\n")
        #         logging_file.close()

        # global min_ch, max_ch, channel_step
        min_ch, max_ch = 21, 44
        channel_step = 1
        num_chs = 4
        channel_tuples = list(zip(np.arange(min_ch, max_ch, num_chs - 1),
                                  np.arange(min_ch + num_chs - 1, max_ch + num_chs - 1, num_chs - 1)))

        t = time.time()

        plot_loop("Loop Centered at 175W", start_file=True)
        plot_loop("Loop Centered at Origin", start_file=True)

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

        def plot_overburden_and_plates(title, ch_step=1, start_file=False):

            log_file_path = sample_files.joinpath(fr"Overburden\{title} log.txt")
            logging_file = open(str(log_file_path), "w+")

            print(f"Plotting {title}")
            logging_file.write(f">>Plotting {title}\n\n")

            maxwell_dir = sample_files.joinpath(r"Overburden\Maxwell\Overburden+Conductor Revised")
            mun_dir = sample_files.joinpath(r"Overburden\MUN\Overburden + plate")

            maxwell_files = list(maxwell_dir.glob("*Only*.TEM"))
            mun_files = list(mun_dir.glob("*Only*.DAT"))
            unique_files = os_sorted(get_unique_files([maxwell_files, mun_files]))

            out_pdf = sample_files.joinpath(fr"Overburden\{title}.PDF")
            footnote = "MUN data filtered using Savitzki-Golay filter"
            count = 0
            with PdfPages(out_pdf) as pdf:
                for stem in unique_files:
                    # stem = stem.title()
                    print(f"Plotting model {stem} ({count + 1}/{len(unique_files)})")
                    format_files = []

                    max_obj = None
                    mun_obj = None

                    max_file = maxwell_dir.joinpath(stem).with_suffix(".TEM")
                    mun_file = mun_dir.joinpath(stem).with_suffix(".DAT")

                    if not max_file.exists():
                        logging_file.write(f"{stem} missing from Maxwell.\n")
                        print(f"{stem} missing from Maxwell.")
                    else:
                        max_obj = TEMFile().parse(max_file)
                        format_files.append(max_obj)

                    if not mun_file.exists():
                        logging_file.write(f"{stem} missing from MUN.\n")
                        print(f"{stem} missing from MUN.")
                    else:
                        mun_obj = MUNFile().parse(mun_file)
                        format_files.append(mun_obj)

                    if not format_files:
                        logging_file.write(f"No files found for {stem}.")
                        print(f"No files found for {stem}.")
                        continue

                    for ch_range in channel_tuples:
                        start_ch, end_ch = ch_range[0], ch_range[1]
                        if ch_range[0] < min_ch:
                            start_ch = min_ch
                        if ch_range[1] > max_ch:
                            end_ch = max_ch
                        print(f"Plotting channel {start_ch} to {end_ch}")

                        if max_obj:
                            plot_obj(ax_dict, max_obj, start_ch, end_ch,
                                     ch_step=channel_step,
                                     station_shift=0,
                                     data_scaling=1e-6,
                                     lc=colors.get("Maxwell")
                                     )

                        if mun_obj:
                            plot_obj(ax_dict, mun_obj, start_ch, end_ch,
                                     ch_step=channel_step,
                                     station_shift=0,
                                     filter=True,
                                     lc=colors.get("MUN")
                                     )

                        format_figure(figure, ax_dict,
                                      f"Overburden Model\n"
                                      f"{stem}\n"
                                      f"{max_obj.ch_times[start_ch - 1]}ms to {max_obj.ch_times[end_ch - 1]}ms",
                                      format_files, start_ch, end_ch,
                                      x_min=max_obj.data.STATION.min(),
                                      x_max=max_obj.data.STATION.max(),
                                      ch_step=channel_step,
                                      incl_legend=True,
                                      incl_legend_ls=True,
                                      incl_legend_colors=True,
                                      style_legend_by='time',
                                      color_legend_by='file',
                                      footnote=footnote)

                        pdf.savefig(figure, orientation='landscape')
                        clear_axes(axes)
                        log_scale([x_ax_log, z_ax_log])
                    count += 1

                if start_file:
                    os.startfile(str(out_pdf))

                runtime = get_runtime(t)
                print(f"{title} runtime: {runtime}")
                logging_file.write(f"{title} runtime: {runtime}\n")
                logging_file.close()

        def plot_contact_effect(title, ch_step=1, start_file=False):
            """Effects of plate contact"""

            def plot_file_contact_effect(filetype, contact_files, separated_files):
                print(F"Plotting effects of plate contact for {filetype} files")

                num_chs = 5
                channel_tuples = list(zip(np.arange(min_ch, max_ch, num_chs - 1),
                                          np.arange(min_ch + num_chs - 1, max_ch + num_chs - 1, num_chs - 1)))

                out_pdf = sample_files.joinpath(fr"Overburden\{title} ({filetype}).PDF")
                count = 0
                with PdfPages(out_pdf) as pdf:
                    format_files = []
                    for con_file, sep_file in zip(contact_files, separated_files):
                        print(f"Plotting {con_file.stem} vs {sep_file.stem} ({count + 1}/{len(contact_files)})")
                        if filetype == "Maxwell":
                            con_obj = TEMFile().parse(con_file)
                            sep_obj = TEMFile().parse(sep_file)
                        else:
                            con_obj = MUNFile().parse(con_file)
                            sep_obj = MUNFile().parse(sep_file)
                        format_files.append(con_obj)
                        format_files.append(sep_obj)

                        for ch_range in channel_tuples:
                            start_ch, end_ch = ch_range[0], ch_range[1]
                            if ch_range[0] < min_ch:
                                start_ch = min_ch
                            if ch_range[1] > max_ch:
                                end_ch = max_ch
                            print(f"Plotting channel {start_ch} to {end_ch}")

                            plot_obj(ax_dict, con_obj, start_ch, end_ch,
                                     ch_step=channel_step,
                                     station_shift=0,
                                     name="Contact",
                                     data_scaling=1e-6 if filetype == "Maxwell" else 1.,
                                     alpha=0.6,
                                     filter=False if filetype == "Maxwell" else True,
                                     ls='-'
                                     )

                            plot_obj(ax_dict, sep_obj, start_ch, end_ch,
                                     ch_step=channel_step,
                                     station_shift=0,
                                     name="Separated",
                                     data_scaling=1e-6 if filetype == "Maxwell" else 1.,
                                     alpha=1.,
                                     filter=False if filetype == "Maxwell" else True,
                                     ls=':'
                                     )

                            model_name = re.search(r"(\d+S Overburden - Plate #\d).*", con_file.stem).group(1)
                            footnote = "MUN data filtered using Savitzki-Golay filter"
                            format_figure(figure, ax_dict,
                                          f"Overburden Model: {filetype} Plate Contact Effect\n"
                                          f"{model_name}\n"
                                          f"{sep_obj.ch_times[start_ch - 1]}ms to {sep_obj.ch_times[end_ch - 1]}ms",
                                          format_files, start_ch, end_ch,
                                          x_min=None,
                                          x_max=None,
                                          ch_step=channel_step,
                                          incl_legend=True,
                                          incl_legend_ls=True,
                                          incl_legend_colors=True,
                                          style_legend_by='line',
                                          color_legend_by='time',
                                          footnote=footnote)

                            pdf.savefig(figure, orientation='landscape')
                            clear_axes(axes)
                            log_scale([x_ax_log, z_ax_log])
                            count += 1

                    if start_file:
                        os.startfile(str(out_pdf))

            def plot_file_contact_differential(max_contact_files, max_separated_files, mun_contact_files, mun_separated_files):
                print(F"Plotting contact effect differential")
                num_chs = 4
                channel_tuples = list(zip(np.arange(min_ch, max_ch, num_chs - 1),
                                          np.arange(min_ch + num_chs - 1, max_ch + num_chs - 1, num_chs - 1)))

                out_pdf = sample_files.joinpath(fr"Overburden\{title} Differential.PDF")
                count = 0
                with PdfPages(out_pdf) as pdf:
                    for max_con_file, max_sep_file, mun_con_file, mun_sep_file in \
                            zip(max_contact_files, max_separated_files, mun_contact_files, mun_separated_files):
                        format_files = []
                        model_name = re.search(r"(\d+S Overburden - Plate #\d).*", max_con_file.stem).group(1)
                        print(f"Plotting {model_name} ({count + 1}/{len(max_contact_files)})")

                        max_con_obj = TEMFile().parse(max_con_file)
                        max_sep_obj = TEMFile().parse(max_sep_file)
                        mun_con_obj = MUNFile().parse(mun_con_file)
                        mun_sep_obj = MUNFile().parse(mun_sep_file)

                        channels = [f'CH{num}' for num in range(min_ch, max_ch - min_ch + 1)]
                        max_diff_obj = copy.deepcopy(max_con_obj)
                        max_diff_obj.data.loc[:, channels] = max_con_obj.data.loc[:, channels] - max_sep_obj.data.loc[:, channels]

                        mun_diff_obj = copy.deepcopy(mun_con_obj)
                        mun_diff_obj.data.loc[:, channels] = mun_con_obj.data.loc[:, channels] - mun_sep_obj.data.loc[:, channels]

                        format_files.append(max_diff_obj)
                        format_files.append(mun_diff_obj)

                        for ch_range in channel_tuples:
                            start_ch, end_ch = ch_range[0], ch_range[1]
                            if ch_range[0] < min_ch:
                                start_ch = min_ch
                            if ch_range[1] > max_ch:
                                end_ch = max_ch
                            print(f"Plotting channel {start_ch} to {end_ch}")

                            plot_obj(ax_dict, max_diff_obj, start_ch, end_ch,
                                     ch_step=channel_step,
                                     station_shift=0,
                                     data_scaling=1e-6,
                                     alpha=1.,
                                     filter=False,
                                     lc=colors.get("Maxwell")
                                     )

                            plot_obj(ax_dict, mun_diff_obj, start_ch, end_ch,
                                     ch_step=channel_step,
                                     station_shift=0,
                                     data_scaling=1.,
                                     alpha=0.9,
                                     filter=True,
                                     lc=colors.get("MUN")
                                     )

                            footnote = "MUN data filtered using Savitzki-Golay filter"
                            format_figure(figure, ax_dict,
                                          f"Overburden Model: Plate Contact Effect Differential (Contact Response - Separated Response)\n"
                                          f"{model_name}\n"
                                          f"{max_diff_obj.ch_times[start_ch - 1]}ms to {max_diff_obj.ch_times[end_ch - 1]}ms",
                                          format_files, start_ch, end_ch,
                                          x_min=None,
                                          x_max=None,
                                          ch_step=channel_step,
                                          incl_legend=True,
                                          incl_legend_ls=True,
                                          incl_legend_colors=True,
                                          style_legend_by='time',
                                          color_legend_by='file',
                                          footnote=footnote)

                            pdf.savefig(figure, orientation='landscape')
                            clear_axes(axes)
                            log_scale([x_ax_log, z_ax_log])
                            count += 1

                    if start_file:
                        os.startfile(str(out_pdf))

            log_file_path = sample_files.joinpath(fr"Overburden\{title} log.txt")
            logging_file = open(str(log_file_path), "w+")

            print(f"Plotting {title}")
            logging_file.write(f">>Plotting {title}\n\n")

            maxwell_dir = sample_files.joinpath(r"Overburden\Maxwell\Overburden+Conductor Revised")
            mun_dir = sample_files.joinpath(r"Overburden\MUN\Overburden + plate")

            max_sep_files = os_sorted(list(maxwell_dir.glob("*Spacing*.TEM")))
            max_con_files = os_sorted(list(maxwell_dir.glob("*Contact*.TEM")))
            mun_sep_files = os_sorted(list(mun_dir.glob("*Spacing*.DAT")))
            mun_con_files = os_sorted(list(mun_dir.glob("*Contact*.DAT")))

            """ Plotting """
            # plot_file_contact_effect("Maxwell", max_con_files, max_sep_files)
            plot_file_contact_effect("MUN", mun_con_files, mun_sep_files)

            plot_file_contact_differential(max_con_files, max_sep_files, mun_con_files, mun_sep_files)

            runtime = get_runtime(t)
            print(f"{title} runtime: {runtime}")
            logging_file.write(f"{title} runtime: {runtime}\n")
            logging_file.close()

            # def plot_maxwell_contact_effect():
            #     print(F">>Plotting Maxwell contact effect ({conductance})")
            #     # Plot the in-contact plate with separated plate for each method
            #     plot_max(axes,
            #              maxwell_comb_sep_file1,
            #              min_ch,
            #              max_ch,
            #              ch_step=ch_step,
            #              name="Separated",
            #              line_color=None,
            #              ls='--',
            #              data_scaling=1e-6,
            #              alpha=1.,
            #              single_file=True,
            #              incl_label=False)
            #     plot_max(axes,
            #              maxwell_comb_con_file1,
            #              min_ch,
            #              max_ch,
            #              ch_step=ch_step,
            #              name="Contact",
            #              line_color=None,
            #              ls='-',
            #              data_scaling=1e-6,
            #              single_file=True,
            #              incl_label=True)
            #     format_figure(figure,
            #                   f"Overburden Models\n"
            #                   f"Maxwell Plate Contact vs Separation [{conductance} Overburden with Plate 1]",
            #                   [maxwell_comb_sep_file1],
            #                   min_ch,
            #                   max_ch,
            #                   incl_legend=True,
            #                   extra_handles=["--", "-"],
            #                   extra_labels=["Separated", "Contact"])
            #     pdf.savefig(figure, orientation='landscape')
            #     clear_axes(axes)
            #     log_scale(x_ax_log, z_ax_log)
            #
            #     # Plot the in-contact plate with separated plate for each method
            #     plot_max(axes,
            #              maxwell_comb_sep_file2,
            #              min_ch,
            #              max_ch,
            #              ch_step=ch_step,
            #              name="Separated",
            #              line_color=None,
            #              ls='--',
            #              data_scaling=1e-6,
            #              alpha=1.,
            #              single_file=True,
            #              incl_label=False)
            #     plot_max(axes,
            #              maxwell_comb_con_file2,
            #              min_ch,
            #              max_ch,
            #              ch_step=ch_step,
            #              name="Contact",
            #              line_color=None,
            #              ls='-',
            #              data_scaling=1e-6,
            #              single_file=True,
            #              incl_label=True)
            #     format_figure(figure,
            #                   f"Overburden Models\n"
            #                   f"Maxwell Plate Contact vs Separation [{conductance} Overburden with Plate 2]",
            #                   [maxwell_comb_sep_file2],
            #                   min_ch,
            #                   max_ch,
            #                   incl_legend=True,
            #                   extra_handles=["--", "-"],
            #                   extra_labels=["Separated", "Contact"])
            #     pdf.savefig(figure, orientation='landscape')
            #     clear_axes(axes)
            #     log_scale(x_ax_log, z_ax_log)
            #
            # def plot_mun_contact_effect():
            #     print(F">>Plotting MUN contact effect ({conductance})")
            #     plot_mun(axes,
            #              mun_comb_sep_file1,
            #              min_ch,
            #              max_ch,
            #              ch_step=ch_step,
            #              name="Separated",
            #              line_color=None,
            #              ls='--',
            #              alpha=1.,
            #              single_file=True,
            #              incl_label=False)
            #     plot_mun(axes,
            #              mun_comb_con_file1,
            #              min_ch,
            #              max_ch,
            #              ch_step=ch_step,
            #              name="Contact",
            #              line_color=None,
            #              ls='-',
            #              single_file=True,
            #              incl_label=True)
            #     format_figure(figure,
            #                   f"Overburden Models\n"
            #                   f"MUN Plate Contact vs Separation [{conductance} Overburden with Plate 1]",
            #                   [mun_comb_sep_file1],
            #                   min_ch,
            #                   max_ch,
            #                   incl_legend=True,
            #                   extra_handles=["--", "-"],
            #                   extra_labels=["Separated", "Contact"])
            #     pdf.savefig(figure, orientation='landscape')
            #     clear_axes(axes)
            #     log_scale(x_ax_log, z_ax_log)
            #
            #     plot_mun(axes,
            #              mun_comb_sep_file2,
            #              min_ch,
            #              max_ch,
            #              ch_step=ch_step,
            #              name="Separated",
            #              line_color=None,
            #              ls='--',
            #              alpha=1.,
            #              single_file=True,
            #              incl_label=False)
            #     plot_mun(axes,
            #              mun_comb_con_file2,
            #              min_ch,
            #              max_ch,
            #              ch_step=ch_step,
            #              name="Contact",
            #              line_color=None,
            #              ls='-',
            #              single_file=True,
            #              incl_label=True)
            #     format_figure(figure,
            #                   f"Overburden Models\n"
            #                   f"MUN Plate Contact vs Separation [{conductance} Overburden with Plate 2]",
            #                   [mun_comb_sep_file2],
            #                   min_ch,
            #                   max_ch,
            #                   incl_legend=True,
            #                   extra_handles=["--", "-"],
            #                   extra_labels=["Separated", "Contact"])
            #     pdf.savefig(figure, orientation='landscape')
            #     clear_axes(axes)
            #     log_scale(x_ax_log, z_ax_log)
            #
            # def plot_differential():
            #     print(F">>Plotting contact differential ({conductance})")
            #     # Calculate the difference between separate and contact plates
            #     plot_max(axes,
            #              maxwell_plate1_diff,
            #              min_ch,
            #              max_ch,
            #              ch_step=ch_step,
            #              name="Maxwell",
            #              line_color=None,
            #              ls=styles.get("Maxwell"),
            #              single_file=True,
            #              incl_label=True,
            #              data_scaling=1e-6,
            #              alpha=1.)
            #
            #     plot_mun(axes,
            #              mun_plate1_diff,
            #              min_ch,
            #              max_ch,
            #              ch_step=ch_step,
            #              name="MUN",
            #              line_color=None,
            #              ls=styles.get("MUN"),
            #              single_file=True,
            #              incl_label=False,
            #              alpha=0.9)
            #
            #     format_figure(figure,
            #                   f"Overburden Models\n"
            #                   f"Separation vs Contact Differential [{conductance} Overburden with Plate 1]",
            #                   [maxwell_plate1_diff,
            #                    mun_plate1_diff],
            #                   min_ch,
            #                   max_ch,
            #                   extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
            #                   extra_labels=["Maxwell", "MUN"])
            #
            #     pdf.savefig(figure, orientation='landscape')
            #     clear_axes(axes)
            #     log_scale(x_ax_log, z_ax_log)
            #
            #     plot_max(axes,
            #              maxwell_plate2_diff,
            #              min_ch,
            #              max_ch,
            #              ch_step=ch_step,
            #              name="Maxwell",
            #              line_color=None,
            #              ls=styles.get("Maxwell"),
            #              single_file=True,
            #              incl_label=True,
            #              data_scaling=1e-6,
            #              alpha=1.)
            #
            #     plot_mun(axes,
            #              mun_plate2_diff,
            #              min_ch,
            #              max_ch,
            #              ch_step=ch_step,
            #              name="MUN",
            #              line_color=None,
            #              ls=styles.get("MUN"),
            #              single_file=True,
            #              incl_label=False,
            #              alpha=0.9)
            #
            #     format_figure(figure,
            #                   f"Overburden Models\n"
            #                   f"Separation vs Contact Differential [{conductance} Overburden with Plate 2]",
            #                   [maxwell_plate2_diff, mun_plate2_diff],
            #                   min_ch,
            #                   max_ch,
            #                   extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
            #                   extra_labels=["Maxwell", "MUN"])
            #
            #     pdf.savefig(figure, orientation='landscape')
            #     clear_axes(axes)
            #     log_scale(x_ax_log, z_ax_log)

            # out_pdf = maxwell_folder.parents[1].joinpath(r"Overburden Model - Effects of Plate Contact.PDF")
            # with PdfPages(out_pdf) as pdf:
            #
            #     for conductance in ["1S", "10S"]:
            #         maxwell_comb_sep_file1 = TEMFile().parse(
            #             Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #1 - 1m Spacing.TEM"))
            #         maxwell_comb_sep_file2 = TEMFile().parse(
            #             Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #2 - 1m Spacing.TEM"))
            #         maxwell_comb_con_file1 = TEMFile().parse(
            #             Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #1 - Contact.TEM"))
            #         maxwell_comb_con_file2 = TEMFile().parse(
            #             Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #2 - Contact.TEM"))
            #
            #         mun_comb_sep_file1 = MUNFile().parse(
            #             Path(mun_folder).joinpath(fr"{conductance}_overburden_plate250_detach_dBdt.DAT"))
            #         mun_comb_sep_file2 = MUNFile().parse(
            #             Path(mun_folder).joinpath(fr"{conductance}_overburden_plate50_detach_dBdt.DAT"))
            #         mun_comb_con_file1 = MUNFile().parse(
            #             Path(mun_folder).joinpath(fr"{conductance}_overburden_plate250_attach_dBdt.DAT"))
            #         mun_comb_con_file2 = MUNFile().parse(
            #             Path(mun_folder).joinpath(fr"{conductance}_overburden_plate50_attach_dBdt.DAT"))
            #
            #         channels = [f'CH{num}' for num in range(1, max_ch - min_ch + 1)]
            #         maxwell_plate1_diff = copy.deepcopy(maxwell_comb_sep_file1)
            #         maxwell_plate2_diff = copy.deepcopy(maxwell_comb_sep_file2)
            #         maxwell_plate1_diff.data.loc[:, channels] = maxwell_comb_con_file1.data.loc[:, channels] - maxwell_comb_sep_file1.data.loc[:, channels]
            #         maxwell_plate2_diff.data.loc[:, channels] = maxwell_comb_con_file2.data.loc[:, channels] - maxwell_comb_sep_file2.data.loc[:, channels]
            #
            #         mun_plate1_diff = copy.deepcopy(mun_comb_sep_file1)
            #         mun_plate2_diff = copy.deepcopy(mun_comb_sep_file2)
            #         mun_plate1_diff.data.loc[:, channels] = mun_comb_con_file1.data.loc[:, channels] - mun_comb_sep_file1.data.loc[:, channels]
            #         mun_plate2_diff.data.loc[:, channels] = mun_comb_con_file2.data.loc[:, channels] - mun_comb_sep_file2.data.loc[:, channels]
            #
            #         plot_maxwell_contact_effect()
            #         plot_mun_contact_effect()
            #         plot_differential()
            # os.startfile(out_pdf)

        def get_overburden_only_file(conductance, filetype):
                print(f"Searching for overburden only file for overburden conductance {conductance} for {filetype}.")
                if filetype == "Maxwell":
                    dir = sample_files.joinpath(r"Overburden\Maxwell\Overburden+Conductor Revised")
                    file = list(dir.glob(f"{conductance}*.TEM"))
                    if len(file) == 0:
                        raise ValueError(F"Overburden file for {filetype} {conductance} not found.")
                    else:
                        print(f"Files found: {', '.join([f.name for f in file])}")
                        return file[0]
                else:
                    dir = sample_files.joinpath(r"Overburden\MUN\Overburden + plate")
                    file = list(dir.glob(f"{conductance}*.DAT"))
                    if len(file) == 0:
                        raise ValueError(F"Overburden file for {filetype} {conductance} not found.")
                    else:
                        print(f"Files found: {', '.join([f.name for f in file])}")
                        return file[0]

        def get_plate_only_file(plate_num, filetype):
            print(f"Searching for plate {plate_num} only for {filetype}.")
            if filetype == "Maxwell":
                dir = sample_files.joinpath(r"Overburden\Maxwell\Overburden+Conductor Revised")
                file = list(dir.glob(f"Plate #{plate_num} Only*.TEM"))
                if len(file) == 0:
                    raise ValueError(F"Plate only file for {filetype} plate {plate_num} not found.")
                else:
                    print(f"Files found:\n{', '.join([f.name for f in file])}")
                    return file[0]
            else:
                dir = sample_files.joinpath(r"Overburden\MUN\Overburden + plate")
                file = list(dir.glob(f"Plate #{plate_num} Only*.DAT"))
                if len(file) == 0:
                    raise ValueError(F"Plate only file for {filetype} plate {plate_num} not found.")
                else:
                    print(f"Files found:\n{', '.join([f.name for f in file])}")
                    return file[0]

        def get_combined_file(conductance, plate_num, filetype):
            print(f"Searching for combined file for overburden conductance {conductance}, plate {plate_num} for {filetype}.")
            if filetype == "Maxwell":
                dir = sample_files.joinpath(r"Overburden\Maxwell\Overburden+Conductor Revised")
                files = list(dir.glob(f"{conductance} Overburden - Plate #{plate_num}*.TEM"))
                if len(files) == 0:
                    raise ValueError(F"Combined model file for {filetype} overburden {conductance} plate {plate_num} not found.")
                else:
                    print(f"Files found:\n{', '.join([f.name for f in files])}")
                    return files
            else:
                dir = sample_files.joinpath(r"Overburden\MUN\Overburden + plate")
                files = list(dir.glob(f"{conductance} Overburden - Plate #{plate_num}*.DAT"))
                if len(files) == 0:
                    raise ValueError(F"Combined model file for {filetype} overburden {conductance} plate {plate_num} not found.")
                else:
                    print(f"Files found:\n{', '.join([f.name for f in files])}")
                    return files

        def plot_residual(title, ch_step=1, start_file=False):

            log_file_path = sample_files.joinpath(fr"Overburden\{title} log.txt")
            logging_file = open(str(log_file_path), "w+")

            print(f"Plotting {title}")
            logging_file.write(f">>Plotting {title}\n\n")

            num_chs = 4
            channel_tuples = list(zip(np.arange(min_ch, max_ch, num_chs - 1),
                                      np.arange(min_ch + num_chs - 1, max_ch + num_chs - 1, num_chs - 1)))

            # out_pdf = sample_files.joinpath(fr"Overburden\{title}.PDF")
            out_pdf = sample_files.joinpath(fr"Overburden\MUN Residual - Savitzky-Golay filter comparison.PDF")

            conductances = ["1S", "10S"]
            plates = ["1", "2"]
            with PdfPages(out_pdf) as pdf:
                for conductance in conductances:
                    for plate in plates:
                        print(f"Plotting residual for plate {plate}, overburden {conductance}")

                        max_ob_file = get_overburden_only_file(conductance, "Maxwell")
                        mun_ob_file = get_overburden_only_file(conductance, "MUN")
                        max_ob_obj = TEMFile().parse(max_ob_file)
                        mun_ob_obj = MUNFile().parse(mun_ob_file)

                        max_plate_file = get_plate_only_file(plate, "Maxwell")
                        mun_plate_file = get_plate_only_file(plate, "MUN")
                        max_plate_obj = TEMFile().parse(max_plate_file)
                        mun_plate_obj = MUNFile().parse(mun_plate_file)

                        max_comb_files = get_combined_file(conductance, plate, "Maxwell")
                        mun_comb_files = get_combined_file(conductance, plate, "MUN")

                        for max_combined_file, mun_combined_file in zip(max_comb_files, mun_comb_files):
                            format_files = []
                            model_name = max_combined_file.stem
                            print(f"Plotting residual for {model_name}")

                            max_comb_obj = TEMFile().parse(max_combined_file)
                            mun_comb_obj = MUNFile().parse(mun_combined_file)

                            max_residual_obj = calc_residual(max_comb_obj, max_ob_obj, max_plate_obj)
                            mun_residual_obj = calc_residual(mun_comb_obj, mun_ob_obj, mun_plate_obj)

                            format_files.append(max_residual_obj)
                            format_files.append(mun_residual_obj)

                            for ch_range in channel_tuples:
                                start_ch, end_ch = ch_range[0], ch_range[1]
                                if ch_range[0] < min_ch:
                                    start_ch = min_ch
                                if ch_range[1] > max_ch:
                                    end_ch = max_ch
                                print(f"Plotting channel {start_ch} to {end_ch}")

                                # """Comparing the filter"""
                                # plot_obj(ax_dict, mun_residual_obj, start_ch, end_ch,
                                #          ch_step=channel_step,
                                #          station_shift=0,
                                #          filter=True,
                                #          name="Filtered",
                                #          alpha=1.,
                                #          ls="-"
                                #          )
                                #
                                # plot_obj(ax_dict, mun_residual_obj, start_ch, end_ch,
                                #          ch_step=channel_step,
                                #          station_shift=0,
                                #          filter=False,
                                #          name="Original",
                                #          alpha=0.5,
                                #          ls=":"
                                #          )
                                #
                                # format_figure(figure, ax_dict,
                                #               f"{title} - Savitzki-Golay Filter\n"
                                #               f"{model_name}\n"
                                #               f"{mun_residual_obj.ch_times[start_ch - 1]}ms to {mun_residual_obj.ch_times[end_ch - 1]}ms",
                                #               format_files, start_ch, end_ch,
                                #               x_min=None,
                                #               x_max=None,
                                #               ch_step=channel_step,
                                #               incl_legend=True,
                                #               incl_legend_ls=True,
                                #               incl_legend_colors=True,
                                #               style_legend_by='line',
                                #               color_legend_by='time',
                                #               footnote="")

                                plot_obj(ax_dict, max_residual_obj, start_ch, end_ch,
                                         ch_step=channel_step,
                                         station_shift=0,
                                         data_scaling=1e-6,
                                         alpha=1.,
                                         filter=False,
                                         lc=colors.get("Maxwell")
                                         )

                                plot_obj(ax_dict, mun_residual_obj, start_ch, end_ch,
                                         ch_step=channel_step,
                                         station_shift=0,
                                         data_scaling=1.,
                                         alpha=0.9,
                                         filter=True,
                                         lc=colors.get("MUN")
                                         )

                                footnote = "MUN data filtered using Savitzki-Golay filter"
                                format_figure(figure, ax_dict,
                                              f"Overburden Model: Residual\n"
                                              f"{model_name}\n"
                                              f"{max_comb_obj.ch_times[start_ch - 1]}ms to {max_comb_obj.ch_times[end_ch - 1]}ms",
                                              format_files, start_ch, end_ch,
                                              x_min=None,
                                              x_max=None,
                                              ch_step=channel_step,
                                              incl_legend=True,
                                              incl_legend_ls=True,
                                              incl_legend_colors=True,
                                              style_legend_by='time',
                                              color_legend_by='file',
                                              footnote=footnote)

                                pdf.savefig(figure, orientation='landscape')
                                clear_axes(axes)
                                log_scale([x_ax_log, z_ax_log])

                if start_file:
                    os.startfile(str(out_pdf))

                runtime = get_runtime(t)
                print(f"{title} runtime: {runtime}")
                logging_file.write(f"{title} runtime: {runtime}\n")
                logging_file.close()

        def plot_residual_percentage(title, ch_step=1, start_file=False):
            log_file_path = sample_files.joinpath(fr"Overburden\{title} log.txt")
            logging_file = open(str(log_file_path), "w+")

            print(f"Plotting {title}")
            logging_file.write(f">>Plotting {title}\n\n")

            num_chs = 4
            channel_tuples = list(zip(np.arange(min_ch, max_ch, num_chs - 1),
                                      np.arange(min_ch + num_chs - 1, max_ch + num_chs - 1, num_chs - 1)))

            out_pdf = sample_files.joinpath(fr"Overburden\{title}.PDF")

            conductances = ["1S", "10S"]
            plates = ["1", "2"]
            with PdfPages(out_pdf) as pdf:
                for conductance in conductances:
                    for plate in plates:
                        print(f"Plotting residual for plate {plate}, overburden {conductance}")

                        max_ob_file = get_overburden_only_file(conductance, "Maxwell")
                        mun_ob_file = get_overburden_only_file(conductance, "MUN")
                        max_ob_obj = TEMFile().parse(max_ob_file)
                        mun_ob_obj = MUNFile().parse(mun_ob_file)

                        max_plate_file = get_plate_only_file(plate, "Maxwell")
                        mun_plate_file = get_plate_only_file(plate, "MUN")
                        max_plate_obj = TEMFile().parse(max_plate_file)
                        mun_plate_obj = MUNFile().parse(mun_plate_file)

                        max_comb_files = get_combined_file(conductance, plate, "Maxwell")
                        mun_comb_files = get_combined_file(conductance, plate, "MUN")

                        for max_combined_file, mun_combined_file in zip(max_comb_files, mun_comb_files):
                            format_files = []
                            model_name = max_combined_file.stem
                            print(f"Plotting residual for {model_name}")

                            max_comb_obj = TEMFile().parse(max_combined_file)
                            mun_comb_obj = MUNFile().parse(mun_combined_file)

                            max_residual_obj = calc_residual(max_comb_obj, max_ob_obj, max_plate_obj)
                            mun_residual_obj = calc_residual(mun_comb_obj, mun_ob_obj, mun_plate_obj)

                            channels = [f"CH{num}" for num in range(min_ch, max_ch + 1)]

                            max_df = max_residual_obj.data.loc[:, channels] * 1e-6
                            mun_df = mun_residual_obj.data.loc[:, channels].astype(float)
                            max_df['STATION'] = max_residual_obj.data.STATION
                            max_df['COMPONENT'] = max_residual_obj.data.COMPONENT
                            mun_df['Station'] = mun_residual_obj.data.Station.astype(float)
                            mun_df['Component'] = mun_residual_obj.data.Component

                            # Drop MUN rows that aren't in Maxwell's DF
                            mun_df.Station = mun_df.Station - 0.2
                            mun_df = mun_df[mun_df.Station.isin(max_df.STATION)]

                            max_df.reset_index(drop=True, inplace=True)
                            mun_df.reset_index(drop=True, inplace=True)

                            # Filter the MUN rows
                            filtered_mun_data = mun_df.loc[:, channels].apply(lambda x: savgol_filter(x, 21, 3),
                                                                              axis=0)

                            # filtered_mun_data = []
                            # for i, row in mun_df.loc[:, channels].iterrows():
                            #     filtered_mun_data.append(savgol_filter(row, 5, 3))

                            # mun_df.loc[:, channels] = pd.DataFrame.from_records(filtered_mun_data.to_numpy(),
                            #                                                     columns=channels)
                            filtered_mun = mun_df.copy()
                            filtered_mun.loc[:, channels] = pd.DataFrame.from_records(filtered_mun_data,
                                                                                      columns=channels)

                            # max_df.sort_values(by=["STATION", "COMPONENT"], inplace=True, ignore_index=True)
                            # mun_df.sort_values(by=["Station", "Component"], inplace=True, ignore_index=True)
                            # df = mun_df.copy()
                            # df.loc[:, channels] = max_df.loc[:, channels] - mun_df.loc[:, channels]

                            format_files.append(max_residual_obj)

                            for ch_range in channel_tuples:
                                start_ch, end_ch = ch_range[0], ch_range[1]
                                if ch_range[0] < min_ch:
                                    start_ch = min_ch
                                if ch_range[1] > max_ch:
                                    end_ch = max_ch
                                print(f"Plotting channel {start_ch} to {end_ch}")

                                plot_obj(ax_dict, mun_df, start_ch, end_ch, ch_times=max_residual_obj.ch_times,
                                         ch_step=channel_step,
                                         station_shift=0,
                                         data_scaling=1,
                                         alpha=1.,
                                         filter=False,
                                         lc="r"
                                         )

                                plot_obj(ax_dict, filtered_mun, start_ch, end_ch, ch_times=max_residual_obj.ch_times,
                                         ch_step=channel_step,
                                         station_shift=0,
                                         data_scaling=1,
                                         alpha=1.,
                                         filter=False,
                                         lc="gray",
                                         )

                                footnote = "MUN data filtered using Savitzki-Golay filter"
                                format_figure(figure, ax_dict,
                                              f"Overburden Model: Residual in percentage\n"
                                              f"{model_name}\n"
                                              f"{max_comb_obj.ch_times[start_ch - 1]}ms to {max_comb_obj.ch_times[end_ch - 1]}ms",
                                              format_files, start_ch, end_ch,
                                              x_min=None,
                                              x_max=None,
                                              ch_step=channel_step,
                                              incl_legend=True,
                                              incl_legend_ls=True,
                                              # incl_legend_colors=True,
                                              style_legend_by='time',
                                              color_legend_by='line',
                                              footnote=footnote,
                                              ylabel="Percentage Difference (%)")

                                pdf.savefig(figure, orientation='landscape')
                                clear_axes(axes)
                                log_scale([x_ax_log, z_ax_log])

                if start_file:
                    os.startfile(str(out_pdf))

                runtime = get_runtime(t)
                print(f"{title} runtime: {runtime}")
                logging_file.write(f"{title} runtime: {runtime}\n")
                logging_file.close()

        def plot_enhancement(title, ch_step=1, start_file=False):

            def calc_enhancement(combined_file, ob_file, plate_file):
                # Works for both MUN and Maxwell
                print(f"Calculating enhancement for {', '.join([f.filepath.name for f in [combined_file, ob_file, plate_file]])}")
                enhance_file = copy.deepcopy(plate_file)
                channels = [f'CH{num}' for num in range(1, len(ob_file.ch_times) + 1)]

                enhance_data = combined_file.data.loc[:, channels] - ob_file.data.loc[:, channels]
                enhance_file.data.loc[:, channels] = enhance_data

                """ Saving to TEM file """
                # if isinstance(plate_file, TEMFile):
                #     global count
                #     if count == 0 or count == 1:
                #         sep = "Separated"
                #     else:
                #         sep = "Contact"
                #     filepath = enhance_file.filepath.parent.with_name(enhance_file.filepath.stem +
                #                                                       f" ({sep}, {conductance} overburden enhancement).TEM")
                #     count += 1
                #     enhance_file.save(filepath=filepath)
                return enhance_file

            log_file_path = sample_files.joinpath(fr"Overburden\{title} log.txt")
            logging_file = open(str(log_file_path), "w+")

            print(f"Plotting {title}")
            logging_file.write(f">>Plotting {title}\n\n")

            num_chs = 4
            channel_tuples = list(zip(np.arange(min_ch, max_ch, num_chs - 1),
                                      np.arange(min_ch + num_chs - 1, max_ch + num_chs - 1, num_chs - 1)))

            out_pdf = sample_files.joinpath(fr"Overburden\{title}.PDF")

            conductances = ["1S", "10S"]
            plates = ["1", "2"]
            with PdfPages(out_pdf) as pdf:
                for conductance in conductances:
                    for plate in plates:
                        print(f"Plotting enhancement for plate {plate}, overburden {conductance}")

                        max_ob_file = get_overburden_only_file(conductance, "Maxwell")
                        mun_ob_file = get_overburden_only_file(conductance, "MUN")
                        max_ob_obj = TEMFile().parse(max_ob_file)
                        mun_ob_obj = MUNFile().parse(mun_ob_file)

                        max_plate_file = get_plate_only_file(plate, "Maxwell")
                        mun_plate_file = get_plate_only_file(plate, "MUN")
                        max_plate_obj = TEMFile().parse(max_plate_file)
                        mun_plate_obj = MUNFile().parse(mun_plate_file)

                        max_comb_files = get_combined_file(conductance, plate, "Maxwell")
                        mun_comb_files = get_combined_file(conductance, plate, "MUN")

                        for max_combined_file, mun_combined_file in zip(max_comb_files, mun_comb_files):
                            format_files = []
                            model_name = max_combined_file.stem
                            print(f"Plotting enhancement for {model_name}")

                            max_comb_obj = TEMFile().parse(max_combined_file)
                            mun_comb_obj = MUNFile().parse(mun_combined_file)

                            max_enhancement_obj = calc_enhancement(max_comb_obj, max_ob_obj, max_plate_obj)
                            mun_enhancement_obj = calc_enhancement(mun_comb_obj, mun_ob_obj, mun_plate_obj)

                            format_files.append(max_enhancement_obj)
                            format_files.append(mun_enhancement_obj)

                            for ch_range in channel_tuples:
                                start_ch, end_ch = ch_range[0], ch_range[1]
                                if ch_range[0] < min_ch:
                                    start_ch = min_ch
                                if ch_range[1] > max_ch:
                                    end_ch = max_ch
                                print(f"Plotting channel {start_ch} to {end_ch}")

                                plot_obj(ax_dict, max_enhancement_obj, start_ch, end_ch,
                                         ch_step=channel_step,
                                         station_shift=0,
                                         data_scaling=1e-6,
                                         alpha=1.,
                                         filter=False,
                                         lc=colors.get("Maxwell")
                                         )

                                plot_obj(ax_dict, mun_enhancement_obj, start_ch, end_ch,
                                         ch_step=channel_step,
                                         station_shift=0,
                                         data_scaling=1.,
                                         alpha=0.9,
                                         filter=True,
                                         lc=colors.get("MUN")
                                         )

                                footnote = "MUN data filtered using Savitzki-Golay filter"
                                format_figure(figure, ax_dict,
                                              f"Overburden Model: Plate Enhancement (Overburden Response Substracted)\n"
                                              f"{model_name}\n"
                                              f"{max_comb_obj.ch_times[start_ch - 1]}ms to {max_comb_obj.ch_times[end_ch - 1]}ms",
                                              format_files, start_ch, end_ch,
                                              x_min=None,
                                              x_max=None,
                                              ch_step=channel_step,
                                              incl_legend=True,
                                              incl_legend_ls=True,
                                              incl_legend_colors=True,
                                              style_legend_by='time',
                                              color_legend_by='file',
                                              footnote=footnote)

                                pdf.savefig(figure, orientation='landscape')
                                clear_axes(axes)
                                log_scale([x_ax_log, z_ax_log])

                if start_file:
                    os.startfile(str(out_pdf))

                runtime = get_runtime(t)
                print(f"{title} runtime: {runtime}")
                logging_file.write(f"{title} runtime: {runtime}\n")
                logging_file.close()

        # def plot_residual(ch_step=1):
        #     """
        #     Compare Maxwell and MUN residuals.
        #     Residual is the effect of mutual induction: combined model - all individual plates.
        #     """
        #
        #     def plot_residual_comparison(ch_step=1):
        #         print(f">> Plotting residual response ({conductance})")
        #
        #         plot_max(axes,
        #                  maxwell_plate_1_residual_sep,
        #                  min_ch,
        #                  max_ch,
        #                  ch_step=ch_step,
        #                  name=f"Maxwell",
        #                  line_color=None,
        #                  ls=styles.get("Maxwell"),
        #                  single_file=True,
        #                  incl_label=True,
        #                  data_scaling=1e-6,
        #                  alpha=1.)
        #
        #         plot_mun(axes,
        #                  mun_plate_1_residual_sep,
        #                  min_ch,
        #                  max_ch,
        #                  ch_step=ch_step,
        #                  name=f"MUN",
        #                  line_color=None,
        #                  ls=styles.get("MUN"),
        #                  single_file=True,
        #                  incl_label=False,
        #                  alpha=1.)
        #
        #         format_figure(figure,
        #                       f"Overburden Models\n"
        #                       f"Residual [{conductance} Overburden with Plate 1, Separated]",
        #                       [maxwell_plate_1_residual_sep,
        #                        mun_plate_1_residual_sep],
        #                       min_ch,
        #                       max_ch,
        #                       incl_legend=True,
        #                       extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
        #                       extra_labels=["Maxwell", "MUN"])
        #
        #         pdf.savefig(figure, orientation='landscape')
        #         clear_axes(axes)
        #         log_scale(x_ax_log, z_ax_log)
        #
        #         """Plate 2 with separation"""
        #         plot_max(axes,
        #                  maxwell_plate_2_residual_sep,
        #                  min_ch,
        #                  max_ch,
        #                  ch_step=ch_step,
        #                  name=f"Maxwell",
        #                  line_color=None,
        #                  ls=styles.get("Maxwell"),
        #                  single_file=True,
        #                  incl_label=True,
        #                  data_scaling=1e-6,
        #                  alpha=1.)
        #
        #         plot_mun(axes,
        #                  mun_plate_2_residual_sep,
        #                  min_ch,
        #                  max_ch,
        #                  ch_step=ch_step,
        #                  name=f"MUN",
        #                  line_color=None,
        #                  ls=styles.get("MUN"),
        #                  single_file=True,
        #                  incl_label=False,
        #                  alpha=1.)
        #
        #         format_figure(figure,
        #                       f"Overburden Models\nResidual [{conductance} Overburden with Plate 2, Separated]",
        #                       [maxwell_plate_2_residual_sep,
        #                        mun_plate_2_residual_sep],
        #                       min_ch,
        #                       max_ch,
        #                       incl_legend=True,
        #                       extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
        #                       extra_labels=["Maxwell", "MUN"])
        #
        #         pdf.savefig(figure, orientation='landscape')
        #         clear_axes(axes)
        #         log_scale(x_ax_log, z_ax_log)
        #
        #         """Plate 1 contact"""
        #         plot_max(axes,
        #                  maxwell_plate_1_residual_con,
        #                  min_ch,
        #                  max_ch,
        #                  ch_step=ch_step,
        #                  name=f"Maxwell",
        #                  line_color=None,
        #                  ls=styles.get("Maxwell"),
        #                  single_file=True,
        #                  incl_label=True,
        #                  data_scaling=1e-6,
        #                  alpha=1.)
        #
        #         plot_mun(axes,
        #                  mun_plate_1_residual_con,
        #                  min_ch,
        #                  max_ch,
        #                  ch_step=ch_step,
        #                  name=f"MUN",
        #                  line_color=None,
        #                  ls=styles.get("MUN"),
        #                  single_file=True,
        #                  incl_label=False,
        #                  alpha=1.)
        #
        #         format_figure(figure,
        #                       f"Overburden Models\nResidual [{conductance} Overburden with Plate 1, Contact]",
        #                       [maxwell_plate_1_residual_con,
        #                        mun_plate_1_residual_con],
        #                       min_ch,
        #                       max_ch,
        #                       incl_legend=True,
        #                       extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
        #                       extra_labels=["Maxwell", "MUN"])
        #
        #         pdf.savefig(figure, orientation='landscape')
        #         clear_axes(axes)
        #         log_scale(x_ax_log, z_ax_log)
        #
        #         """Plate 2 contact"""
        #         plot_max(axes,
        #                  maxwell_plate_2_residual_con,
        #                  min_ch,
        #                  max_ch,
        #                  ch_step=ch_step,
        #                  name=f"Maxwell",
        #                  line_color=None,
        #                  ls=styles.get("Maxwell"),
        #                  single_file=True,
        #                  incl_label=True,
        #                  data_scaling=1e-6,
        #                  alpha=1.)
        #
        #         plot_mun(axes,
        #                  mun_plate_2_residual_con,
        #                  min_ch,
        #                  max_ch,
        #                  ch_step=ch_step,
        #                  name=f"MUN",
        #                  line_color=None,
        #                  ls=styles.get("MUN"),
        #                  single_file=True,
        #                  incl_label=False,
        #                  alpha=1.)
        #
        #         format_figure(figure,
        #                       f"Overburden Models\nResidual [{conductance} Overburden with Plate 2, Contact]",
        #                       [maxwell_plate_2_residual_con,
        #                        mun_plate_2_residual_con],
        #                       min_ch,
        #                       max_ch,
        #                       incl_legend=True,
        #                       extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
        #                       extra_labels=["Maxwell", "MUN"])
        #
        #         pdf.savefig(figure, orientation='landscape')
        #         clear_axes(axes)
        #         log_scale(x_ax_log, z_ax_log)
        #
        #     """Compare residual/mutual inductance"""
        #     out_pdf = maxwell_folder.parents[1].joinpath(r"Overburden Model - Residual.PDF")
        #     with PdfPages(out_pdf) as pdf:
        #
        #         for conductance in ["1S", "10S"]:
        #             maxwell_ob_file = TEMFile().parse(Path(maxwell_folder).joinpath(fr"{conductance} Overburden Only - 50m.TEM"))
        #             mun_ob_file = MUNFile().parse(Path(mun_folder).joinpath(fr"overburden_{conductance}_V1000m_dBdt.DAT"))
        #
        #             maxwell_comb_sep_file1 = TEMFile().parse(
        #                 Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #1 - 1m Spacing.TEM"))
        #             maxwell_comb_sep_file2 = TEMFile().parse(
        #                 Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #2 - 1m Spacing.TEM"))
        #             maxwell_comb_con_file1 = TEMFile().parse(
        #                 Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #1 - Contact.TEM"))
        #             maxwell_comb_con_file2 = TEMFile().parse(
        #                 Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #2 - Contact.TEM"))
        #
        #             mun_comb_sep_file1 = MUNFile().parse(
        #                 Path(mun_folder).joinpath(fr"{conductance}_overburden_plate250_detach_dBdt.DAT"))
        #             mun_comb_sep_file2 = MUNFile().parse(
        #                 Path(mun_folder).joinpath(fr"{conductance}_overburden_plate50_detach_dBdt.DAT"))
        #             mun_comb_con_file1 = MUNFile().parse(
        #                 Path(mun_folder).joinpath(fr"{conductance}_overburden_plate250_attach_dBdt.DAT"))
        #             mun_comb_con_file2 = MUNFile().parse(
        #                 Path(mun_folder).joinpath(fr"{conductance}_overburden_plate50_attach_dBdt.DAT"))
        #
        #             maxwell_plate_1_residual_sep = calc_residual(maxwell_comb_sep_file1, maxwell_ob_file, maxwell_plate1_file)
        #             maxwell_plate_2_residual_sep = calc_residual(maxwell_comb_sep_file2, maxwell_ob_file, maxwell_plate2_file)
        #             maxwell_plate_1_residual_con = calc_residual(maxwell_comb_con_file1, maxwell_ob_file, maxwell_plate1_file)
        #             maxwell_plate_2_residual_con = calc_residual(maxwell_comb_con_file2, maxwell_ob_file, maxwell_plate2_file)
        #             mun_plate_1_residual_sep = calc_residual(mun_comb_sep_file1, mun_ob_file, mun_plate1_file)
        #             mun_plate_2_residual_sep = calc_residual(mun_comb_sep_file2, mun_ob_file, mun_plate2_file)
        #             mun_plate_1_residual_con = calc_residual(mun_comb_con_file1, mun_ob_file, mun_plate1_file)
        #             mun_plate_2_residual_con = calc_residual(mun_comb_con_file2, mun_ob_file, mun_plate2_file)
        #
        #             plot_residual_comparison(ch_step=ch_step)
        #     os.startfile(out_pdf)

        # def analyze_residual(ch_step=1):
        #     """
        #     Compare Maxwell and MUN residuals.
        #     Residual is the effect of mutual induction: combined model - all individual plates.
        #     """
        #
        #     def get_residual_diff(maxwell_file, mun_file):
        #         channels = [f'CH{num}' for num in range(min_ch, max_ch + 1)]
        #         diff_file = maxwell_file
        #         diff_data = pd.DataFrame()
        #
        #         for component in diff_file.data.COMPONENT.unique():
        #             maxwell_station_filt = maxwell_file.data.STATION.astype(float).isin(mun_file.data.Station.astype(float) - 0.2)
        #             mun_station_filt = (mun_file.data.Station.astype(float) - 0.2).isin(maxwell_file.data.STATION.astype(float))
        #             maxwell_filt = (maxwell_file.data.COMPONENT == component) & (maxwell_station_filt)
        #             mun_filt = (mun_file.data.Component == component) & (mun_station_filt)
        #
        #             maxwell_data = maxwell_file.data[maxwell_filt].reset_index(drop=True).loc[:, channels] * 1e-6
        #             mun_data = mun_file.data[mun_filt].reset_index(drop=True).loc[:, channels]
        #
        #             diff = maxwell_data.abs() - mun_data.abs()
        #             diff.insert(0, "STATION", maxwell_file.data[maxwell_filt].reset_index(drop=True).STATION)
        #             diff.insert(0, "COMPONENT", maxwell_file.data[maxwell_filt].reset_index(drop=True).COMPONENT)
        #             diff_data = diff_data.append(diff)
        #         return diff_data
        #
        #     def get_residual_percent(combined_file, residual_file):
        #         diff_data = pd.DataFrame()
        #         for component in combined_file.data.COMPONENT.unique():
        #             model_filt = combined_file.data.COMPONENT == component
        #             residual_filt = residual_file.data.COMPONENT == component
        #             model_data = combined_file.data[model_filt].loc[:, channels].reset_index(drop=True)
        #             residual_data = residual_file.data[residual_filt].loc[:, channels].reset_index(drop=True)
        #             diff = residual_data / model_data * 100
        #
        #             if isinstance(combined_file, TEMFile):
        #                 diff.insert(0, "STATION", combined_file.data.STATION)
        #                 diff.insert(0, "COMPONENT", component)
        #                 diff_data = diff_data.append(diff)
        #             else:
        #                 diff.insert(0, "STATION", combined_file.data.Station)
        #                 diff.insert(0, "COMPONENT", component)
        #                 diff_data = diff_data.append(diff)
        #         return diff_data
        #
        #     def plot_residual_differential(ch_step=1):
        #         print(f">> Plotting residual difference ({conductance})")
        #
        #         """Separated"""
        #         # Use a maxwell file to make plotting simpler
        #         # diff_data = get_residual_diff(maxwell_plate_1_residual_sep, mun_plate_1_residual_sep)
        #
        #         diff_data = get_residual_percent(maxwell_comb_sep_file1, maxwell_plate_1_residual_sep)
        #         # diff_data = get_residual_percent(mun_comb_sep_file1, mun_plate_1_residual_sep)
        #
        #         plot_df(axes, diff_data, maxwell_plate_1_residual_sep, ch_step=ch_step)
        #         # plot_df(axes, diff_data, mun_comb_sep_file1, ch_step=ch_step)
        #         # plot_df(axes, maxwell_comb_sep_file1.data, mun_comb_sep_file1, ch_step=ch_step,
        #         #         ls="-")
        #         # plot_df(axes, maxwell_plate_1_residual_sep.data, mun_comb_sep_file1, ch_step=ch_step,
        #         #         ls=":")
        #         format_figure(figure,
        #                       f"Overburden Models\n"
        #                       f"Residual Differential [{conductance} Overburden with Plate 1, Separated]",
        #                       [maxwell_plate_1_residual_sep,
        #                        mun_plate_1_residual_sep],
        #                       min_ch,
        #                       max_ch,
        #                       incl_legend=True,
        #                       )
        #
        #         pdf.savefig(figure, orientation='landscape')
        #         clear_axes(axes)
        #         log_scale(x_ax_log, z_ax_log)
        #
        #         # """Plate 2 with separation"""
        #         # plot_maxwell(axes,
        #         #              maxwell_plate_2_residual_sep,
        #         #              min_ch,
        #         #              max_ch,
        #         #              ch_step=ch_step,
        #         #              name=f"Maxwell",
        #         #              line_color=None,
        #         #              line_style=styles.get("Maxwell"),
        #         #              single_file=True,
        #         #              incl_label=True,
        #         #              data_scaling=1e-6,
        #         #              alpha=1.)
        #         #
        #         # plot_mun(axes,
        #         #          mun_plate_2_residual_sep,
        #         #          min_ch,
        #         #          max_ch,
        #         #          ch_step=ch_step,
        #         #          name=f"MUN",
        #         #          line_color=None,
        #         #          line_style=styles.get("MUN"),
        #         #          single_file=True,
        #         #          incl_label=False,
        #         #          alpha=1.)
        #         #
        #         # format_figure(figure,
        #         #               f"Overburden Models\nResidual [{conductance} Overburden with Plate 2, Separated]",
        #         #               [maxwell_plate_2_residual_sep,
        #         #                mun_plate_2_residual_sep],
        #         #               min_ch,
        #         #               max_ch,
        #         #               incl_legend=True,
        #         #               extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
        #         #               extra_labels=["Maxwell", "MUN"])
        #         #
        #         # pdf.savefig(figure, orientation='landscape')
        #         # clear_axes(axes)
        #         # log_scale(x_ax_log, z_ax_log)
        #         #
        #         # """Plate 1 contact"""
        #         # plot_maxwell(axes,
        #         #              maxwell_plate_1_residual_con,
        #         #              min_ch,
        #         #              max_ch,
        #         #              ch_step=ch_step,
        #         #              name=f"Maxwell",
        #         #              line_color=None,
        #         #              line_style=styles.get("Maxwell"),
        #         #              single_file=True,
        #         #              incl_label=True,
        #         #              data_scaling=1e-6,
        #         #              alpha=1.)
        #         #
        #         # plot_mun(axes,
        #         #          mun_plate_1_residual_con,
        #         #          min_ch,
        #         #          max_ch,
        #         #          ch_step=ch_step,
        #         #          name=f"MUN",
        #         #          line_color=None,
        #         #          line_style=styles.get("MUN"),
        #         #          single_file=True,
        #         #          incl_label=False,
        #         #          alpha=1.)
        #         #
        #         # format_figure(figure,
        #         #               f"Overburden Models\nResidual [{conductance} Overburden with Plate 1, Contact]",
        #         #               [maxwell_plate_1_residual_con,
        #         #                mun_plate_1_residual_con],
        #         #               min_ch,
        #         #               max_ch,
        #         #               incl_legend=True,
        #         #               extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
        #         #               extra_labels=["Maxwell", "MUN"])
        #         #
        #         # pdf.savefig(figure, orientation='landscape')
        #         # clear_axes(axes)
        #         # log_scale(x_ax_log, z_ax_log)
        #         #
        #         # """Plate 2 contact"""
        #         # plot_maxwell(axes,
        #         #              maxwell_plate_2_residual_con,
        #         #              min_ch,
        #         #              max_ch,
        #         #              ch_step=ch_step,
        #         #              name=f"Maxwell",
        #         #              line_color=None,
        #         #              line_style=styles.get("Maxwell"),
        #         #              single_file=True,
        #         #              incl_label=True,
        #         #              data_scaling=1e-6,
        #         #              alpha=1.)
        #         #
        #         # plot_mun(axes,
        #         #          mun_plate_2_residual_con,
        #         #          min_ch,
        #         #          max_ch,
        #         #          ch_step=ch_step,
        #         #          name=f"MUN",
        #         #          line_color=None,
        #         #          line_style=styles.get("MUN"),
        #         #          single_file=True,
        #         #          incl_label=False,
        #         #          alpha=1.)
        #         #
        #         # format_figure(figure,
        #         #               f"Overburden Models\nResidual [{conductance} Overburden with Plate 2, Contact]",
        #         #               [maxwell_plate_2_residual_con,
        #         #                mun_plate_2_residual_con],
        #         #               min_ch,
        #         #               max_ch,
        #         #               incl_legend=True,
        #         #               extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
        #         #               extra_labels=["Maxwell", "MUN"])
        #         #
        #         # pdf.savefig(figure, orientation='landscape')
        #         # clear_axes(axes)
        #         # log_scale(x_ax_log, z_ax_log)
        #
        #     """Compare residual/mutual inductance"""
        #     out_pdf = maxwell_folder.parents[1].joinpath(r"Overburden Model - Residual Analysis.PDF")
        #     with PdfPages(out_pdf) as pdf:
        #
        #         for conductance in ["1S", "10S"]:
        #             maxwell_ob_file = TEMFile().parse(Path(maxwell_folder).joinpath(fr"{conductance} Overburden Only - 50m.TEM"))
        #             mun_ob_file = MUNFile().parse(Path(mun_folder).joinpath(fr"overburden_{conductance}_V1000m_dBdt.DAT"))
        #
        #             maxwell_comb_sep_file1 = TEMFile().parse(
        #                 Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #1 - 1m Spacing.TEM"))
        #             maxwell_comb_sep_file2 = TEMFile().parse(
        #                 Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #2 - 1m Spacing.TEM"))
        #             maxwell_comb_con_file1 = TEMFile().parse(
        #                 Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #1 - Contact.TEM"))
        #             maxwell_comb_con_file2 = TEMFile().parse(
        #                 Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #2 - Contact.TEM"))
        #
        #             mun_comb_sep_file1 = MUNFile().parse(
        #                 Path(mun_folder).joinpath(fr"{conductance}_overburden_plate250_detach_dBdt.DAT"))
        #             mun_comb_sep_file2 = MUNFile().parse(
        #                 Path(mun_folder).joinpath(fr"{conductance}_overburden_plate50_detach_dBdt.DAT"))
        #             mun_comb_con_file1 = MUNFile().parse(
        #                 Path(mun_folder).joinpath(fr"{conductance}_overburden_plate250_attach_dBdt.DAT"))
        #             mun_comb_con_file2 = MUNFile().parse(
        #                 Path(mun_folder).joinpath(fr"{conductance}_overburden_plate50_attach_dBdt.DAT"))
        #
        #             maxwell_plate_1_residual_sep = calc_residual(maxwell_comb_sep_file1, maxwell_ob_file, maxwell_plate1_file)
        #             maxwell_plate_2_residual_sep = calc_residual(maxwell_comb_sep_file2, maxwell_ob_file, maxwell_plate2_file)
        #             maxwell_plate_1_residual_con = calc_residual(maxwell_comb_con_file1, maxwell_ob_file, maxwell_plate1_file)
        #             maxwell_plate_2_residual_con = calc_residual(maxwell_comb_con_file2, maxwell_ob_file, maxwell_plate2_file)
        #             mun_plate_1_residual_sep = calc_residual(mun_comb_sep_file1, mun_ob_file, mun_plate1_file)
        #             mun_plate_2_residual_sep = calc_residual(mun_comb_sep_file2, mun_ob_file, mun_plate2_file)
        #             mun_plate_1_residual_con = calc_residual(mun_comb_con_file1, mun_ob_file, mun_plate1_file)
        #             mun_plate_2_residual_con = calc_residual(mun_comb_con_file2, mun_ob_file, mun_plate2_file)
        #
        #             plot_residual_differential(ch_step=ch_step)
        #     os.startfile(out_pdf)
        #
        # def plot_enhancement(ch_step=1):
        #     """
        #     Compare Maxwell and MUN plate enhancement
        #     """
        #
        #     def plot_enhancement_comparison(ch_step=1):
        #         """Compare plate enhancement"""
        #         plot_max(axes,
        #                  maxwell_plate_1_enhance_sep,
        #                  min_ch,
        #                  max_ch,
        #                  ch_step=ch_step,
        #                  name=f"Maxwell",
        #                  line_color=None,
        #                  ls=styles.get("Maxwell"),
        #                  single_file=True,
        #                  incl_label=True,
        #                  data_scaling=1e-6,
        #                  alpha=1.)
        #
        #         plot_mun(axes,
        #                  mun_plate_1_enhance_sep,
        #                  min_ch,
        #                  max_ch,
        #                  ch_step=ch_step,
        #                  name=f"MUN",
        #                  line_color=None,
        #                  ls=styles.get("MUN"),
        #                  single_file=True,
        #                  incl_label=False,
        #                  alpha=1.)
        #
        #         format_figure(figure,
        #                       f"Overburden Models\n"
        #                       f"Plate Enhancement (Overburden Response Substracted) [{conductance} Overburden with Plate 1, Separated]",
        #                       [maxwell_plate_1_enhance_sep,
        #                        mun_plate_1_enhance_sep],
        #                       min_ch,
        #                       max_ch,
        #                       incl_legend=True,
        #                       extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
        #                       extra_labels=["Maxwell", "MUN"])
        #
        #         pdf.savefig(figure, orientation='landscape')
        #         clear_axes(axes)
        #         log_scale(x_ax_log, z_ax_log)
        #
        #         """Plate 2 with separation"""
        #         plot_max(axes,
        #                  maxwell_plate_2_enhance_sep,
        #                  min_ch,
        #                  max_ch,
        #                  ch_step=ch_step,
        #                  name=f"Maxwell",
        #                  line_color=None,
        #                  ls=styles.get("Maxwell"),
        #                  single_file=True,
        #                  incl_label=True,
        #                  data_scaling=1e-6,
        #                  alpha=1.)
        #
        #         plot_mun(axes,
        #                  mun_plate_2_enhance_sep,
        #                  min_ch,
        #                  max_ch,
        #                  ch_step=ch_step,
        #                  name=f"MUN",
        #                  line_color=None,
        #                  ls=styles.get("MUN"),
        #                  single_file=True,
        #                  incl_label=False,
        #                  alpha=1.)
        #
        #         format_figure(figure,
        #                       f"Overburden Models\n"
        #                       f"Plate Enhancement (Overburden Response Substracted) [{conductance} Overburden with Plate 2, Separated]",
        #                       [maxwell_plate_2_enhance_sep,
        #                        mun_plate_2_enhance_sep],
        #                       min_ch,
        #                       max_ch,
        #                       incl_legend=True,
        #                       extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
        #                       extra_labels=["Maxwell", "MUN"])
        #
        #         pdf.savefig(figure, orientation='landscape')
        #         clear_axes(axes)
        #         log_scale(x_ax_log, z_ax_log)
        #
        #         """Plate 1 contact"""
        #         plot_max(axes,
        #                  maxwell_plate_1_enhance_con,
        #                  min_ch,
        #                  max_ch,
        #                  ch_step=ch_step,
        #                  name=f"Maxwell",
        #                  line_color=None,
        #                  ls=styles.get("Maxwell"),
        #                  single_file=True,
        #                  incl_label=True,
        #                  data_scaling=1e-6,
        #                  alpha=1.)
        #
        #         plot_mun(axes,
        #                  mun_plate_1_enhance_con,
        #                  min_ch,
        #                  max_ch,
        #                  ch_step=ch_step,
        #                  name=f"MUN",
        #                  line_color=None,
        #                  ls=styles.get("MUN"),
        #                  single_file=True,
        #                  incl_label=False,
        #                  alpha=1.)
        #
        #         format_figure(figure,
        #                       f"Overburden Models\n"
        #                       f"Plate Enhancement (Overburden Response Substracted) [{conductance} Overburden with Plate 1, Contact]",
        #                       [maxwell_plate_1_enhance_con,
        #                        mun_plate_1_enhance_con],
        #                       min_ch,
        #                       max_ch,
        #                       incl_legend=True,
        #                       extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
        #                       extra_labels=["Maxwell", "MUN"])
        #
        #         pdf.savefig(figure, orientation='landscape')
        #         clear_axes(axes)
        #         log_scale(x_ax_log, z_ax_log)
        #
        #         """Plate 2 contact"""
        #         plot_max(axes,
        #                  maxwell_plate_2_enhance_con,
        #                  min_ch,
        #                  max_ch,
        #                  ch_step=ch_step,
        #                  name=f"Maxwell",
        #                  line_color=None,
        #                  ls=styles.get("Maxwell"),
        #                  single_file=True,
        #                  incl_label=True,
        #                  data_scaling=1e-6,
        #                  alpha=1.)
        #
        #         plot_mun(axes,
        #                  mun_plate_2_enhance_con,
        #                  min_ch,
        #                  max_ch,
        #                  ch_step=ch_step,
        #                  name=f"MUN",
        #                  line_color=None,
        #                  ls=styles.get("MUN"),
        #                  single_file=True,
        #                  incl_label=False,
        #                  alpha=1.)
        #
        #         format_figure(figure,
        #                       f"Overburden Models\n"
        #                       f"Plate Enhancement (Overburden Response Substracted) [{conductance} Overburden with Plate 2, Contact]",
        #                       [maxwell_plate_2_enhance_con,
        #                        mun_plate_2_enhance_con],
        #                       min_ch,
        #                       max_ch,
        #                       incl_legend=True,
        #                       extra_handles=[styles.get("Maxwell"), styles.get("MUN")],
        #                       extra_labels=["Maxwell", "MUN"])
        #
        #         pdf.savefig(figure, orientation='landscape')
        #         clear_axes(axes)
        #         log_scale(x_ax_log, z_ax_log)
        #
        #     out_pdf = maxwell_folder.parents[1].joinpath(r"Overburden Model - Enhancement.PDF")
        #     with PdfPages(out_pdf) as pdf:
        #
        #         for conductance in ["1S", "10S"]:
        #             print(f">> Plotting enhancement ({conductance})")
        #             maxwell_ob_file = TEMFile().parse(Path(maxwell_folder).joinpath(fr"{conductance} Overburden Only - 50m.TEM"))
        #             mun_ob_file = MUNFile().parse(Path(mun_folder).joinpath(fr"overburden_{conductance}_V1000m_dBdt.DAT"))
        #
        #             maxwell_comb_sep_file1 = TEMFile().parse(
        #                 Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #1 - 1m Spacing.TEM"))
        #             maxwell_comb_sep_file2 = TEMFile().parse(
        #                 Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #2 - 1m Spacing.TEM"))
        #             maxwell_comb_con_file1 = TEMFile().parse(
        #                 Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #1 - Contact.TEM"))
        #             maxwell_comb_con_file2 = TEMFile().parse(
        #                 Path(maxwell_folder).joinpath(fr"{conductance} Overburden - Plate #2 - Contact.TEM"))
        #
        #             mun_comb_sep_file1 = MUNFile().parse(
        #                 Path(mun_folder).joinpath(fr"{conductance}_overburden_plate250_detach_dBdt.DAT"))
        #             mun_comb_sep_file2 = MUNFile().parse(
        #                 Path(mun_folder).joinpath(fr"{conductance}_overburden_plate50_detach_dBdt.DAT"))
        #             mun_comb_con_file1 = MUNFile().parse(
        #                 Path(mun_folder).joinpath(fr"{conductance}_overburden_plate250_attach_dBdt.DAT"))
        #             mun_comb_con_file2 = MUNFile().parse(
        #                 Path(mun_folder).joinpath(fr"{conductance}_overburden_plate50_attach_dBdt.DAT"))
        #
        #             global count
        #             count = 0
        #             maxwell_plate_1_enhance_sep = calc_enhancement(maxwell_comb_sep_file1, maxwell_ob_file, maxwell_plate1_file)
        #             maxwell_plate_2_enhance_sep = calc_enhancement(maxwell_comb_sep_file2, maxwell_ob_file, maxwell_plate2_file)
        #             maxwell_plate_1_enhance_con = calc_enhancement(maxwell_comb_con_file1, maxwell_ob_file, maxwell_plate1_file)
        #             maxwell_plate_2_enhance_con = calc_enhancement(maxwell_comb_con_file2, maxwell_ob_file, maxwell_plate2_file)
        #             mun_plate_1_enhance_sep = calc_enhancement(mun_comb_sep_file1, mun_ob_file, mun_plate1_file)
        #             mun_plate_2_enhance_sep = calc_enhancement(mun_comb_sep_file2, mun_ob_file, mun_plate2_file)
        #             mun_plate_1_enhance_con = calc_enhancement(mun_comb_con_file1, mun_ob_file, mun_plate1_file)
        #             mun_plate_2_enhance_con = calc_enhancement(mun_comb_con_file2, mun_ob_file, mun_plate2_file)
        #
        #             # plot_enhancement_comparison(ch_step=ch_step)
        #     # os.startfile(out_pdf)

        figure, ((x_ax, z_ax), (x_ax_log, z_ax_log)) = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='none')
        ax_dict = {"X": (x_ax, x_ax_log), "Y": (None, None), "Z": (z_ax, z_ax_log)}
        axes = [x_ax, z_ax, x_ax_log, z_ax_log]
        figure.set_size_inches((11 * 1.33 * 1.33, 8.5 * 1.33))
        log_scale([x_ax_log, z_ax_log])

        min_ch, max_ch = 21, 44
        channel_step = 1
        num_chs = 4
        channel_tuples = list(zip(np.arange(min_ch, max_ch, num_chs - 1),
                                  np.arange(min_ch + num_chs - 1, max_ch + num_chs - 1, num_chs - 1)))

        t = time.time()

        # plot_overburden_and_plates("Overburden Model - Plates & Overburden Only",
        #                            ch_step=channel_step,
        #                            start_file=True)
        # plot_contact_effect("Overburden Model - Plate Contact Effect",
        #                     ch_step=channel_step,
        #                     start_file=True)
        plot_residual("Overburden Model - Residual",
                      ch_step=channel_step,
                      start_file=True)
        # plot_residual_percentage("Overburden Model - Residual (%)",
        #                          ch_step=channel_step,
        #                          start_file=True)
        # plot_enhancement("Overburden Model - Enhancement",
        #                  ch_step=channel_step,
        #                  start_file=True)

    def plot_bentplate():

        def plot_model(model_name, title, pdf, max_dir, mun_dir, logging_file, residual=False, ylabel=''):
            max_file = max_dir.joinpath(model_name).with_suffix(".TEM")
            mun_file = mun_dir.joinpath(model_name).with_suffix(".DAT")
            format_files = []

            max_obj = None
            single_plate_max_obj = None
            mun_obj = None

            if not max_file.exists():
                logging_file.write(f"{model_name} missing from Maxwell.\n")
                print(f"{model_name} missing from Maxwell.")
            else:
                max_obj = TEMFile().parse(max_file)
                if residual is True:
                    max_obj = get_residual_file(max_obj, max_dir, model_name)
                format_files.append(max_obj)

            if model_name in single_plate_files:
                print(f"Searching for single plate version of {model_name}")
                single_plate_max_file = max_dir.joinpath(model_name + " (single plate)").with_suffix(".TEM")
                if not single_plate_max_file.exists():
                    print(f'Cannot find {max_dir.joinpath(model_name + " (single plate)").with_suffix(".TEM")}')
                else:
                    print(f'Found {max_dir.joinpath(model_name + " (single plate)").with_suffix(".TEM")}')
                    single_plate_max_obj = TEMFile().parse(single_plate_max_file)
                    if residual is True:
                        single_plate_max_obj = get_residual_file(single_plate_max_obj, max_dir, model_name)

            if not mun_file.exists():
                logging_file.write(f"{model_name} missing from MUN.\n")
                print(f"{model_name} missing from MUN.")
            else:
                mun_obj = MUNFile().parse(mun_file)
                if residual is True:
                    mun_obj = get_residual_file(mun_obj, mun_dir, model_name)
                format_files.append(mun_obj)

            if not format_files:
                logging_file.write(f"No files found for {model_name}.")
                print(f"No files found for {model_name}.")
                return

            for ch_range in channel_tuples:
                footnote = []
                start_ch, end_ch = ch_range[0], ch_range[1]
                if ch_range[0] < min_ch:
                    start_ch = min_ch
                if ch_range[1] > max_ch:
                    end_ch = max_ch
                print(f"Plotting channel {start_ch} to {end_ch}")

                if max_obj:
                    plot_obj(ax_dict, max_obj, start_ch, end_ch,
                             ch_step=channel_step,
                             station_shift=-200,
                             data_scaling=1e-6,
                             name="Maxwell",
                             lc=colors.get("Maxwell")
                             )

                    if single_plate_max_obj is not None:
                        plot_obj(ax_dict, single_plate_max_obj, start_ch, end_ch,
                                 ch_step=channel_step,
                                 station_shift=-200,
                                 data_scaling=1e-6,
                                 name="Maxwell (Single Plate)",
                                 lc="dimgray",
                                 )

                if mun_obj:
                    plot_obj(ax_dict, mun_obj, start_ch, end_ch,
                             ch_step=channel_step,
                             station_shift=0,
                             filter=residual,
                             name="MUN",
                             lc=colors.get("MUN")
                             )

                if residual:
                    footnote.append("MUN data filtered using Savitzki-Golay filter.")

                format_figure(figure, ax_dict,
                              f"Bent and Multiple Plates: {title}\n"
                              f"{model_name}\n"
                              f"{format_files[0].ch_times[start_ch - 1]}ms to {format_files[0].ch_times[end_ch - 1]}ms",
                              format_files, start_ch, end_ch,
                              # x_min=max_obj.data.STATION.min(),
                              # x_max=max_obj.data.STATION.max(),
                              ch_step=channel_step,
                              incl_legend=True,
                              incl_legend_ls=True,
                              incl_legend_colors=True,
                              style_legend_by='time',
                              color_legend_by='line',
                              footnote='    '.join(footnote))

                pdf.savefig(figure, orientation='landscape')
                clear_axes(axes)
                log_scale([x_ax_log, y_ax_log, z_ax_log])

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
            #
            # if mun_file.is_file():
            #     print(f"Plotting {mun_file.name}.")
            #     dat_file = MUNFile().parse(mun_file)
            #     if residual is True:
            #         dat_file = get_residual_file(dat_file, mun_folder, single_plot_order)
            #     files.append(dat_file)
            #     log_scale(x_ax_log, z_ax_log)
            #     plot_mun(axes, dat_file, min_ch, max_ch,
            #              ch_step=channel_step,
            #              name=mun_file.name,
            #              ls=styles.get("MUN"),
            #              station_shift=0,
            #              data_scaling=1.,
            #              y_min=y_min,
            #              y_max=y_max)
            #     if file_times is None:
            #         file_times = dat_file.ch_times
            # else:
            #     print(F"MUN file {mun_file.name} not found.")
            #     logging_file.write(F"MUN file {mun_file.name} not found.\n")
            #
            # name = "Multiple and Bent Plate Models\n" + title + " " + model_name
            # format_figure(figure, name, files, min_ch, max_ch,
            #               ch_step=channel_step,
            #               b_field=False,
            #               incl_legend=True,
            #               incl_legend_ls=True,
            #               legend_times=file_times,
            #               ylabel=ylabel)
            # pdf.savefig(figure, orientation='landscape')
            # clear_axes(axes)

        def plot_individual_plates(title, start_file=False):
            """ Plot individual plates"""
            log_file_path = sample_files.joinpath(fr"Bent and multiple plates\{title} log.txt")
            logging_file = open(str(log_file_path), "w+")

            print(f"Plotting {title}")
            logging_file.write(f">>Plotting {title}\n\n")

            out_pdf = sample_files.joinpath(fr"Bent and Multiple Plates\{title}.PDF")

            count = 0
            with PdfPages(out_pdf) as pdf:
                for model in single_plot_order:
                    print(f"Plotting model {model} ({count + 1}/{len(single_plot_order)})")
                    plot_model(model, title, pdf, max_folder_100S, mun_folder_100S, logging_file)
                    count += 1
            if start_file:
                os.startfile(out_pdf)

        def plot_combined_plates(title, start_file=False):
            """ Plot combined plate models"""
            log_file_path = sample_files.joinpath(fr"Bent and multiple plates\{title} log.txt")
            logging_file = open(str(log_file_path), "w+")

            print(f"Plotting {title}")
            logging_file.write(f">>Plotting {title}\n\n")

            out_pdf = sample_files.joinpath(fr"Bent and Multiple Plates\{title}.PDF")
            count = 0
            with PdfPages(out_pdf) as pdf:
                for model in combined_plot_order:
                    print(f"Plotting model {model} ({count + 1}/{len(combined_plot_order)})")
                    plot_model(model, title, pdf, max_folder_100S, mun_folder_100S, logging_file)
                    count += 1
            if start_file:
                os.startfile(out_pdf)

        def plot_contact_effect(title, start_file=False):
            """ Plot effect of connected vs separated plates"""

            def get_objects(models, folder, filetype, logging_file):
                objects = []
                for i, model_name in enumerate(models):
                    file = folder.joinpath(model_name).with_suffix(extensions.get(filetype)[1:])

                    if not file.exists():
                        logging_file.write(f"{model_name} missing from {filetype}.\n")
                        print(f"{model_name} missing from {filetype}.")
                        objects.append(None)
                    else:
                        if filetype == "Maxwell":
                            obj = TEMFile().parse(file)
                        elif filetype == "MUN":
                            obj = MUNFile().parse(file)
                        elif filetype == "PLATE":
                            obj = PlateFFile().parse(file)
                        elif filetype == "IRAP":
                            obj = IRAPFile().parse(file)
                        else:
                            raise ValueError(f"{filetype} is not a valid filetype.")
                        objects.append(obj)

                return objects

            def plot_filetype(filetype, folder):

                log_file_path = sample_files.joinpath(fr"Bent and multiple plates\{title} ({filetype}) log.txt")
                logging_file = open(str(log_file_path), "w+")

                print(f"Plotting {title} ({filetype})")
                logging_file.write(f">>Plotting {title}\n\n")

                out_pdf = sample_files.joinpath(fr"Bent and Multiple Plates\{title} ({filetype}).PDF")
                count = 0
                with PdfPages(out_pdf) as pdf:
                    # Find unique plate combinations
                    combinations = sorted(np.unique([re.sub(r"\D", "", p) for p in combined_plot_order]),
                                          key=lambda x: len(x))

                    for plates in combinations:
                        print(f"Plotting plates {plates} ({count + 1}/{len(combinations)})")
                        # Find all models for the plates
                        models = [model for model in combined_plot_order if re.sub(r"\D", "", model) == plates]
                        objects = get_objects(models, folder, filetype, logging_file)
                        # color_cycle = cm.jet(np.linspace(0, 1, len(models) + 1))
                        color_cycle = ["b", "g", "r", "c", "m"]

                        if len (models) < 2:
                            print(f"Skipping plates {plates} as there aren't enough models")
                            continue

                        for ch_range in channel_tuples:
                            footnote = []
                            start_ch, end_ch = ch_range[0], ch_range[1]
                            if ch_range[0] < min_ch:
                                start_ch = min_ch
                            if ch_range[1] > max_ch:
                                end_ch = max_ch
                            print(f"Plotting channel {start_ch} to {end_ch}")

                            for i, (model, obj) in enumerate(zip(models, objects)):
                                model = models[i]

                                if obj is None:
                                    print(f"Skipping {model}")
                                    continue

                                plot_obj(ax_dict, obj, start_ch, end_ch,
                                         ch_step=channel_step,
                                         station_shift=-200,
                                         data_scaling=1e-6,
                                         alpha=0.6,
                                         name=model,
                                         lc=color_cycle[i]
                                         )

                            format_figure(figure, ax_dict,
                                          f"Bent and Multiple Plates: {title} ({filetype})\n"
                                          f"Plates: {', '.join(sorted(list(plates)))}\n"
                                          f"{objects[0].ch_times[start_ch - 1]}ms to {objects[0].ch_times[end_ch - 1]}ms",
                                          objects, start_ch, end_ch,
                                          # x_min=max_obj.data.STATION.min(),
                                          # x_max=max_obj.data.STATION.max(),
                                          ch_step=channel_step,
                                          incl_legend=True,
                                          incl_legend_ls=True,
                                          incl_legend_colors=True,
                                          style_legend_by='time',
                                          color_legend_by='line',
                                          footnote="")

                            pdf.savefig(figure, orientation='landscape')
                            clear_axes(axes)
                            log_scale([x_ax_log, y_ax_log, z_ax_log])

                        count += 1

                if start_file:
                    os.startfile(out_pdf)

            plot_filetype("MUN", mun_folder_100S)
            # plot_filetype("Maxwell", max_folder_100S)

        def plot_residual(title, start_file=False):
            """ Plot residuals """
            log_file_path = sample_files.joinpath(fr"Bent and multiple plates\{title} log.txt")
            logging_file = open(str(log_file_path), "w+")

            print(f"Plotting {title}")
            logging_file.write(f">>Plotting {title}\n\n")

            out_pdf = sample_files.joinpath(fr"Bent and Multiple Plates\{title}.PDF")
            count = 0
            with PdfPages(out_pdf) as pdf:
                combined_files = [f for f in combined_plot_order if len(f) > 1]
                for model in combined_files:
                    print(f"Plotting model {model} ({count + 1}/{len(combined_files)})")
                    plot_model(model, title, pdf, max_folder_100S, mun_folder_100S, logging_file,
                               residual=True,
                               ylabel="Residual (nT/s)")
                    count += 1

            if start_file:
                os.startfile(out_pdf)

        def plot_varying_conductances(title, start_file):
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
                "1+4+5+3+6@400S": "1+4+5+3+6",
                "(1+4+5)_(3+6)@200S": "(1+4+5)_(3+6)"
            }

            log_file_path = sample_files.joinpath(fr"Bent and multiple plates\{title} log.txt")
            logging_file = open(str(log_file_path), "w+")

            print(f"Plotting {title}")
            logging_file.write(f">>Plotting {title}\n\n")

            out_pdf = sample_files.joinpath(fr"Bent and Multiple Plates\{title}.PDF")
            count = 0
            with PdfPages(out_pdf) as pdf:
                for model in models.keys():
                    print(f"Plotting model {model} ({count + 1}/{len(models)})")
                    plot_model(model, title, pdf, max_folder_varying, mun_folder_varying, logging_file)
                    count += 1
            if start_file:
                os.startfile(out_pdf)

        max_folder_100S = sample_files.joinpath(r"Bent and Multiple Plates\Maxwell\Revised\100S Plates")
        mun_folder_100S = sample_files.joinpath(r"Bent and Multiple Plates\MUN\100S Plates")
        max_folder_varying = sample_files.joinpath(r"Bent and Multiple Plates\Maxwell\Revised\Various Conductances")
        mun_folder_varying = sample_files.joinpath(r"Bent and Multiple Plates\MUN\Various Conductances")
        assert all([max_folder_100S.exists(), mun_folder_100S.exists(), max_folder_varying.exists(), mun_folder_varying.exists()]), \
            "One or more of the folders doesn't exist."

        figure, ((x_ax, z_ax), (x_ax_log, z_ax_log)) = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='none')
        y_ax, y_ax_log = None, None
        ax_dict = {"X": (x_ax, x_ax_log), "Y": (y_ax, y_ax_log), "Z": (z_ax, z_ax_log)}
        axes = [x_ax, y_ax, z_ax, x_ax_log, y_ax_log, z_ax_log]
        figure.set_size_inches((11 * 1.33 * 1.33, 8.5 * 1.33))
        log_scale([x_ax_log, y_ax_log, z_ax_log])

        min_ch, max_ch = 21, 44
        channel_step = 1
        num_chs = 4
        channel_tuples = list(zip(np.arange(min_ch, max_ch, num_chs - 1),
                                  np.arange(min_ch + num_chs - 1, max_ch + num_chs - 1, num_chs - 1)))

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

        single_plate_files = [
            "1+2",
            "2+3",
            "1+2+3",
            "(1+2+3)_6",
        ]

        t = time.time()
        # plot_individual_plates("Individual Plates", start_file=True)
        plot_combined_plates("Combined Plates", start_file=True)
        # plot_contact_effect("Contact Effect", start_file=True)
        # plot_residual("Residual", start_file=True)
        # plot_varying_conductances("Varying Conductances", start_file=True)

        runtime = get_runtime(t)
        print(f"Bent and multiple plates runtime: {runtime}")

    def test_savgol_filter():
        print(f"Plotting Savgol comparison plots")
        figure, ((x_ax, y_ax, z_ax), (x_ax_log, y_ax_log, z_ax_log)) = plt.subplots(nrows=2, ncols=3, sharex='all', sharey='none')
        ax_dict = {"X": (x_ax, x_ax_log), "Y": (y_ax, y_ax_log), "Z": (z_ax, z_ax_log)}
        axes = [x_ax, y_ax, z_ax, x_ax_log, y_ax_log, z_ax_log]
        figure.set_size_inches((11 * 1.33 * 1.33, 8.5 * 1.33))
        log_scale([x_ax_log, y_ax_log, z_ax_log])

        min_ch, max_ch = 21, 44
        channel_step = 1
        num_chs = 4
        channel_tuples = list(zip(np.arange(min_ch, max_ch, num_chs - 1),
                                  np.arange(min_ch + num_chs - 1, max_ch + num_chs - 1, num_chs - 1)))

        directories = list(sample_files.iterdir())
        dir_count = 0
        t = time.time()
        for dir in directories:
            if not dir.is_dir():
                dir_count += 1
                continue

            print(f"Directory: {dir.stem} ({dir_count + 1}/{len(directories)})")
            dat_files = dir.rglob("*.DAT")
            mun_files = [f for f in dat_files if "MUN" in f.parts and "Crone" not in f.stem]
            print(f"MUN files found in {dir.stem}: {len(mun_files)}")
            if not mun_files:
                print(f"Skipping {dir.stem} as no valid MUN files were found.")
                dir_count += 1
                continue

            out_pdf = sample_files.joinpath(fr"{dir.stem} Savitzki-Golay filter.PDF")
            mun_files = mun_files[:]
            count = 0
            with PdfPages(out_pdf) as pdf:
                for mun_file in mun_files:
                    print(f"Plotting file {mun_file.stem} ({count + 1}/{len(mun_files)})")
                    obj = MUNFile().parse(mun_file)
                    format_files = [obj]

                    for ch_range in channel_tuples:
                        start_ch, end_ch = ch_range[0], ch_range[1]
                        if ch_range[0] < min_ch:
                            start_ch = min_ch
                        if ch_range[1] > max_ch:
                            end_ch = max_ch
                        # print(f"Plotting channel {start_ch} to {end_ch}")

                        plot_obj(ax_dict, obj, start_ch, end_ch,
                                 ch_step=channel_step,
                                 station_shift=0,
                                 filter=True,
                                 name="Filtered",
                                 alpha=1.,
                                 ls="-"
                                 )

                        plot_obj(ax_dict, obj, start_ch, end_ch,
                                 ch_step=channel_step,
                                 station_shift=0,
                                 filter=False,
                                 name="Original",
                                 alpha=0.5,
                                 ls=":"
                                 )

                        format_figure(figure, ax_dict,
                                      f"Savitzki-Golay Filter: {dir.stem}\n"
                                      f"{mun_file.stem}\n"
                                      f"{obj.ch_times[start_ch - 1]}ms to {obj.ch_times[end_ch - 1]}ms",
                                      format_files, start_ch, end_ch,
                                      x_min=None,
                                      x_max=None,
                                      ch_step=channel_step,
                                      incl_legend=True,
                                      incl_legend_ls=True,
                                      incl_legend_colors=True,
                                      style_legend_by='line',
                                      color_legend_by='time',
                                      footnote="")

                        pdf.savefig(figure, orientation='landscape')
                        clear_axes(axes)
                        log_scale([x_ax_log, y_ax_log, z_ax_log])
                    count += 1

            dir_count += 1
            os.startfile(str(out_pdf))
        print(f"Plotting complete after {get_runtime(t)}.")

    def plot_flat_plates():

        def plot_model(model_name, title, pdf, max_dir, mun_dir, logging_file, ylabel=''):
            max_obj = None
            mun_obj = None
            format_files = []

            if max_dir is not None:
                max_file = max_dir.joinpath(model_name).with_suffix(".TEM")
                if not max_file.exists():
                    logging_file.write(f"{model_name} missing from Maxwell.\n")
                    print(f"{model_name} missing from Maxwell.")
                else:
                    max_obj = TEMFile().parse(max_file)
                    # if residual is True:
                    #     max_obj = get_residual_file(max_obj, max_dir, model_name)
                    format_files.append(max_obj)

            if mun_dir is not None:
                mun_file = mun_dir.joinpath(model_name).with_suffix(".DAT")
                if not mun_file.exists():
                    logging_file.write(f"{model_name} missing from MUN.\n")
                    print(f"{model_name} missing from MUN.")
                else:
                    mun_obj = MUNFile().parse(mun_file)
                    # if residual is True:
                    #     mun_obj = get_residual_file(mun_obj, mun_dir, model_name)
                    format_files.append(mun_obj)

            if not format_files:
                logging_file.write(f"No files found for {model_name}.")
                print(f"No files found for {model_name}.")
                return

            for ch_range in channel_tuples:
                footnote = []
                start_ch, end_ch = ch_range[0], ch_range[1]
                if ch_range[0] < min_ch:
                    start_ch = min_ch
                if ch_range[1] > max_ch:
                    end_ch = max_ch
                print(f"Plotting channel {start_ch} to {end_ch}")

                if max_obj:
                    plot_obj(ax_dict, max_obj, start_ch, end_ch,
                             ch_step=channel_step,
                             station_shift=0,
                             data_scaling=1e-6,
                             name="Maxwell",
                             lc=colors.get("Maxwell")
                             )

                if mun_obj:
                    plot_obj(ax_dict, mun_obj, start_ch, end_ch,
                             ch_step=channel_step,
                             station_shift=0,
                             filter=False,
                             name="MUN",
                             lc=colors.get("MUN")
                             )

                # if residual:
                #     footnote.append("MUN data filtered using Savitzki-Golay filter.")

                format_figure(figure, ax_dict,
                              f"{title}\n"
                              f"{model_name}\n"
                              f"{format_files[0].ch_times[start_ch - 1]}ms to {format_files[0].ch_times[end_ch - 1]}ms",
                              format_files, start_ch, end_ch,
                              # x_min=max_obj.data.STATION.min(),
                              # x_max=max_obj.data.STATION.max(),
                              ch_step=channel_step,
                              incl_legend=True,
                              incl_legend_ls=True,
                              incl_legend_colors=True,
                              style_legend_by='time',
                              color_legend_by='line',
                              footnote='    '.join(footnote))

                pdf.savefig(figure, orientation='landscape')
                clear_axes(axes)
                log_scale([x_ax_log, y_ax_log, z_ax_log])

        def plot_model1(title, start_file=False):
            log_file_path = sample_files.joinpath(fr"Flat Plates\{title} log.txt")
            logging_file = open(str(log_file_path), "w+")

            print(f"Plotting {title}")
            logging_file.write(f">>Plotting {title}\n\n")

            out_pdf = sample_files.joinpath(fr"Flat Plates\{title}.PDF")

            model_files = [
                "Center Hole - Dual Large Plates",
                "Center Hole - Large Plate",
                "Edge Hole - Dual Large Plates",
                "Edge Hole - Large Plate",
                "In Hole - Dual Large Plate (250m east of west edge)",
                "In Hole - Large Plate (250m east of west edge)",
                "Off Hole - Dual Large Plates (100m west of edge)",
                "Off Hole - Large Plate (100m west of edge)"
            ]

            count = 0
            with PdfPages(out_pdf) as pdf:
                for model in model_files:
                    print(f"Plotting model {model} ({count + 1}/{len(model_files)})")
                    plot_model(model, title, pdf, max_model1_dir, None, logging_file)
                    count += 1
            if start_file:
                os.startfile(out_pdf)

        def plot_model2(title, start_file=False):
            log_file_path = sample_files.joinpath(fr"Flat Plates\{title} log.txt")
            logging_file = open(str(log_file_path), "w+")

            print(f"Plotting {title}")
            logging_file.write(f">>Plotting {title}\n\n")

            out_pdf = sample_files.joinpath(fr"Flat Plates\{title}.PDF")

            model_files = [
                "Center Hole - Dual Small Plates",
                "Center Hole - Small Plate",
                "Edge Hole - Dual Small Plates",
                "Edge Hole - Small Plate",
                "Off Hole - Dual Small Plates (50m west of edge)",
                "Off Hole - Small Plate (50m west of edge)"
            ]

            count = 0
            with PdfPages(out_pdf) as pdf:
                for model in model_files:
                    print(f"Plotting model {model} ({count + 1}/{len(model_files)})")
                    plot_model(model, title, pdf, max_model2_dir, None, logging_file)
                    count += 1
            if start_file:
                os.startfile(out_pdf)

        max_model1_dir = sample_files.joinpath(r"Flat Plates\Maxwell\100x100 loop - 1000x1000 plate")
        max_model2_dir = sample_files.joinpath(r"Flat Plates\Maxwell\400x400 loop - 50x50 plate")
        assert all([max_model1_dir.exists(), max_model2_dir.exists()]), \
            "One or more of the folders doesn't exist."

        figure, ((x_ax, z_ax), (x_ax_log, z_ax_log)) = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='none')
        y_ax, y_ax_log = None, None
        ax_dict = {"X": (x_ax, x_ax_log), "Y": (y_ax, y_ax_log), "Z": (z_ax, z_ax_log)}
        axes = [x_ax, y_ax, z_ax, x_ax_log, y_ax_log, z_ax_log]
        figure.set_size_inches((11 * 1.33 * 1.33, 8.5 * 1.33))
        log_scale([x_ax_log, y_ax_log, z_ax_log])

        min_ch, max_ch = 21, 44
        channel_step = 1
        num_chs = 4
        channel_tuples = list(zip(np.arange(min_ch, max_ch, num_chs - 1),
                                  np.arange(min_ch + num_chs - 1, max_ch + num_chs - 1, num_chs - 1)))

        t = time.time()
        plot_model1("100x100 loop - 1000x1000 plate", start_file=True)
        plot_model2("400x400 loop - 50x50 plate", start_file=True)

        runtime = get_runtime(t)
        print(f"Flat plates runtime: {runtime}")

    # TODO Change "MUN" to "EM3D"
    # plot_aspect_ratio()
    # plot_two_way_induction()
    # plot_run_on_effect()
    # plot_infinite_thin_sheet()
    # plot_infinite_half_sheet()
    # plot_overburden()
    # plot_bentplate()
    # test_savgol_filter()
    plot_flat_plates()

    # tester = TestRunner()
    # tester.show()
    # tester.add_row(sample_files.joinpath(r"Two-way induction\300x100\100S\MUN"), file_type="MUN")
    # # tester.add_row(sample_files.joinpath(r"Two-way induction\300x100\100S\Maxwell"), file_type="Maxwell")
    # tester.test_name_edit.setText("Testing this bullshit")
    # tester.output_filepath_edit.setText(str(sample_files.joinpath(
    #     r"Two-way induction\300x100\100S\MUN\MUN plotting test.PDF")))
    # tester.print_pdf()

    # os.startfile(log_file_path)
    app.exec_()
