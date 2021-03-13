import sys
import os
import pickle
import io
import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from itertools import zip_longest
from natsort import natsorted, os_sorted

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.pyplot import cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator

from src.file_types.fem_file import FEMFile, FEMTab
from src.file_types.tem_file import TEMFile, TEMTab
from src.file_types.platef_file import PlateFFile, PlateFTab
from src.file_types.peter_file import PeterFile
from src.file_types.mun_file import MUNFile, MUNTab
from PyQt5 import (QtCore, QtGui, uic)
from PyQt5.QtWidgets import (QMainWindow, QApplication, QMessageBox, QFrame, QErrorMessage, QFileDialog,
                             QTableWidgetItem, QScrollArea, QSpinBox, QHBoxLayout, QLabel, QInputDialog, QLineEdit,
                             QProgressDialog, QWidget, QHeaderView, QPushButton, QColorDialog)

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
        self.actionConvert_Peter_File.triggered.connect(self.open_peter_converter)
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

    def open_peter_converter(self):
        """
        Convert a Peter File (.txt) to a .csv file for each model inside. Saves to the same directory.
        """
        default_path = str(Path(__file__).parents[1].joinpath('sample_files'))
        dlg = QFileDialog()
        peter_file, ext = dlg.getOpenFileName(self, "Select Peter File", default_path, "Peter Files (*.txt)")

        if peter_file:
            parser = PeterFile()
            parser.convert(peter_file)

    def add_row(self, folderpath=None, file_type=None):
        """Add a row to the table"""
        # File type options with extensions
        options = {"Maxwell": "*.TEM", "MUN": "*.DAT", "Peter": "*.XYZ", "PLATE": "*.DAT"}
        colors = {"Maxwell": '#0000FF', "MUN": '##00FF00', "Peter": "#2C2C2C", "PLATE": '#FF0000'}

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
        options = {"Maxwell": "*.TEM", "MUN": "*.DAT", "Peter": "*.XYZ", "PLATE": "*.DAT"}
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

            size = 8  # For scatter point size

            for ind, ch in enumerate(plotting_channels):
                if ind == 0:
                    label = f"{file.filepath.stem} (Maxwell)"

                    if min_ch == max_ch:
                        self.footnote += f"Maxwell file plotting channel {min_ch + 1} ({file.ch_times[max_ch]:.3f}ms).  "
                    else:
                        self.footnote += f"Maxwell file plotting channels {min_ch + 1}-{max_ch + 1}" \
                            f" ({file.ch_times[min_ch]:.3f}ms-{file.ch_times[max_ch]:.3f}ms).  "
                else:
                    label = None

                x = comp_data.STATION.astype(float) + properties['station_shift']
                y = comp_data.loc[:, ch].astype(float) * properties['scaling']

                if len(x) == 1:
                    style = 'o'
                    self.ax.scatter(x, y,
                                    color=color,
                                    marker=style,
                                    s=size,
                                    alpha=properties['alpha'],
                                    label=label,
                                    zorder=1)

                else:
                    # style = '--' if 'Q' in freq else '-'
                    self.ax.plot(x, y,
                                 color=color,
                                 alpha=properties['alpha'],
                                 label=label,
                                 zorder=1)

                size += 2

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

            size = 8  # For scatter point size

            for ind, ch in enumerate(plotting_channels):
                if ind == 0:
                    label = f"{file.filepath.stem} (PLATE)"

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

                if len(x) == 1:
                    style = 'o'
                    self.ax.scatter(x, y,
                                    color=color,
                                    marker=style,
                                    s=size,
                                    alpha=properties['alpha'],
                                    label=label,
                                    zorder=2)

                else:
                    # style = '--' if 'Q' in freq else '-'
                    self.ax.plot(x, y,
                                 color=color,
                                 alpha=properties['alpha'],
                                 # lw=count / 100,
                                 label=label,
                                 zorder=2)

                size += 2

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
                    self.msg.warning(self, "Different Units", f"The units of {file.filepath.name} are different then"
                    f"the existing units ({file.units} vs {self.units})")

            channels = [f'{num}' for num in range(1, len(file.ch_times) + 1)]
            min_ch = properties['ch_start'] - 1
            max_ch = min(properties['ch_end'] - 1, len(channels) - 1)
            plotting_channels = channels[min_ch: max_ch + 1]

            comp_data = file.data[file.data.COMPONENT == component]

            if comp_data.empty:
                print(f"No {component} data in {file.filepath.name}.")
                return

            size = 8  # For scatter point size

            for ind, ch in enumerate(plotting_channels):
                # If coloring by channel, uses the rainbow color iterator and the label is the channel number.
                if ind == 0:
                    label = f"{file.filepath.stem} (MUN)"
                else:
                    label = None

                x = comp_data.STATION.astype(float) + properties['station_shift']
                y = comp_data.loc[:, ch].astype(float) * properties['scaling']

                if len(x) == 1:
                    style = 'o'
                    self.ax.scatter(x, y,
                                    color=color,
                                    marker=style,
                                    s=size,
                                    alpha=properties['alpha'],
                                    label=label,
                                    zorder=3)

                else:
                    self.ax.plot(x, y,
                                 color=color,
                                 alpha=properties['alpha'],
                                 label=label,
                                 zorder=3)

                size += 2

        def plot_peter(filepath, component):
            raise NotImplementedError(F"Peter files haven't been implement yet.")

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

        # self.ax2.get_yaxis().set_visible(False)
        # self.ax.tick_params(axis='y', labelcolor='k')
        progress = QProgressDialog("Processing...", "Cancel", 0, int(num_files_found))
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setWindowTitle("Printing Profiles")
        progress.show()

        if self.fixed_range_cbox.isChecked():
            y_range = np.array(get_fixed_range())

        count = 0
        progress.setValue(count)
        progress.setLabelText("Printing Profile Plots")
        with PdfPages(pdf_filepath) as pdf:
            for maxwell_file, mun_file, peter_file, plate_file in list(zip_longest(*plotting_files.values(),
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
                    if peter_file:
                        plot_peter(peter_file, component)
                    if plate_file:
                        plot_plate(plate_file, component)

                    # Set the labels
                    self.ax.set_xlabel(f"Station")
                    self.ax.set_ylabel(f"{component} Component Response\n({self.units})")
                    self.ax.set_title(self.test_name_edit.text())

                    if self.custom_stations_cbox.isChecked():
                        self.ax.set_xlim([self.station_start_sbox.value(), self.station_end_sbox.value()])
                    if self.fixed_range_cbox.isChecked():
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

                    # plt.show()
                    pdf.savefig(self.figure, orientation='landscape')
                    self.ax.clear()

                count += 1
                progress.setValue(count)

        os.startfile(pdf_filepath)

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

            label = f"{file.filepath.stem} (Maxwell)"

            self.footnote += f"Maxwell file plotting station {station}.  "

            # style = '--' if 'Q' in freq else '-'
            self.ax.plot(x, decay,
                         color=color,
                         alpha=properties['alpha'],
                         label="Linear-scale",
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

        def plot_peter(filepath, component):
            raise NotImplementedError("Peter decay plots not implemented yet.")

        # self.ax2.get_yaxis().set_visible(True)
        self.ax.tick_params(axis='y', labelcolor='blue')
        progress = QProgressDialog("Processing...", "Cancel", 0, int(num_files_found))
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setWindowTitle("Printing Decays")
        progress.show()
        count = 0

        with PdfPages(pdf_filepath) as pdf:
            for maxwell_file, mun_file, peter_file, plate_file in list(zip_longest(*plotting_files.values(),
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
                    if peter_file:
                        plot_peter(peter_file, component)
                    if plate_file:
                        plot_plate(plate_file, component)

                    # Set the labels
                    self.ax.set_xlabel(f"Time (ms)")
                    self.ax.set_ylabel(f"{component} Component Response\n({self.units})")
                    self.ax.set_title(f"{self.test_name_edit.text()} - {maxwell_file.stem}")

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

                    # plt.show()
                    pdf.savefig(self.figure, orientation='landscape')
                    self.ax.clear()
                    # self.ax2.clear()
                    # self.ax2.set_yscale('symlog', subs=list(np.arange(2, 10, 1)))
                    # self.ax2.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))

                count += 1
                progress.setValue(count)

        os.startfile(pdf_filepath)

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

        def plot_peter(filepath, component):
            raise NotImplementedError("Peter run-on not implemented yet.")

        with PdfPages(pdf_filepath) as pdf:
            if plotting_files['Maxwell']:
                plot_maxwell_decays(plotting_files['Maxwell'], pdf)
            if plotting_files['MUN']:
                plot_mun(plotting_files['MUN'], pdf)
            if plotting_files['Peter']:
                plot_peter(plotting_files['Peter'], pdf)
            if plotting_files['PLATE']:
                plot_plate(plotting_files['PLATE'], pdf)

        os.startfile(pdf_filepath)

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

        with PdfPages(pdf_filepath) as pdf:
            if plotting_files['Maxwell']:
                plot_maxwell_convergence(plotting_files['Maxwell'], pdf)

        os.startfile(pdf_filepath)

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
            os.startfile(output_filepath)

        output_filepath = self.output_filepath_edit.text()

        if plotting_files['Maxwell']:
            tabulate_maxwell_convergence(plotting_files['Maxwell'])

    def print_pdf(self):
        """Create the PDF"""
        if self.table.rowCount() == 0:
            return

        pdf_filepath = self.output_filepath_edit.text()
        if not pdf_filepath:
            self.msg.information(self, "Error", f"PDF output path must not be empty.")
            return

        # Ensure there are equal number of files found for each file type
        num_files_found = self.table.item(0, self.header_labels.index("Files Found")).text()
        for row in range(self.table.rowCount()):
            if self.table.item(row, self.header_labels.index("Files Found")).text() != num_files_found:
                self.msg.critical(self, "Error", "Each file type must have equal number of files.")
                return

        t0 = time.time()

        # Create a dictionary of files to plot
        plotting_files = {"Maxwell": [], "MUN": [], "Peter": [], "PLATE": []}
        for row in range(self.table.rowCount()):
            files = os_sorted(self.opened_files[row])
            file_type = self.table.item(row, self.header_labels.index('File Type')).text()

            for file in files:
                plotting_files[file_type].append(file)

        if not any(plotting_files.values()):
            raise ValueError("No plotting files found.")

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

        print(f"Process complete after {(time.time() - t0) / 60:02.0f}:{(time.time() - t0) % 60:02.0f}")


if __name__ == '__main__':
    import time

    app = QApplication(sys.argv)

    sample_files = Path(__file__).parents[1].joinpath('sample_files')

    # fem_file = sample_files.joinpath(r'Maxwell files\FEM\Horizontal Plate 100S Normalized.fem')
    # tem_file = sample_files.joinpath(r'Aspect ratio\Maxwell\5x150A.TEM')

    tester = TestRunner()
    tester.show()

    def plot_aspect_ratio():
        tester.plot_profiles_rbtn.setChecked(True)
        tester.test_name_edit.setText(r"Aspect Ratio")
        tester.output_filepath_edit.setText(str(sample_files.joinpath(
            r"Aspect Ratio\Aspect Ratio.PDF")))
        # tester.fixed_range_cbox.setChecked(True)

        # Maxwell
        maxwell_dir = sample_files.joinpath(r"Aspect Ratio\Maxwell\2m stations")
        tester.add_row(str(maxwell_dir), "Maxwell")
        tester.table.item(0, 2).setText("0.000001")
        tester.table.item(0, 4).setText("21")
        tester.table.item(0, 5).setText("44")

        # Plate
        plate_dir = sample_files.joinpath(r"Aspect Ratio\PLATE\2m stations")
        tester.add_row(str(plate_dir), "PLATE")
        tester.table.item(1, 7).setText("0.5")

        # Peter
        plate_dir = sample_files.joinpath(r"Aspect Ratio\Peter")
        tester.add_row(str(plate_dir), "Peter")
        tester.table.item(1, 7).setText("0.5")

        tester.print_pdf()

    def plot_two_way_induction():
        tester.plot_profiles_rbtn.setChecked(True)
        tester.test_name_edit.setText(r"Two-Way Induction - 300mx100m Plate")
        tester.output_filepath_edit.setText(str(sample_files.joinpath(
            r"Two-Way Induction\300x100 Two-Way Induction (100S, Fixed Y).PDF")))
        tester.fixed_range_cbox.setChecked(True)
        # tester.output_filepath_edit.setText(str(sample_files.joinpath(
        #     r"Two-Way Induction\300x100 Two-Way Induction (100S).PDF")))

        # Maxwell
        maxwell_dir = sample_files.joinpath(r"Two-Way Induction\300x100\100S\Maxwell")
        tester.add_row(str(maxwell_dir), "Maxwell")
        tester.table.item(0, 2).setText("0.000001")
        tester.table.item(0, 4).setText("21")
        tester.table.item(0, 5).setText("44")

        # Plate
        plate_dir = sample_files.joinpath(r"Two-Way Induction\300x100\100S\PLATE")
        tester.add_row(str(plate_dir), "PLATE")
        tester.table.item(1, 7).setText("0.5")

        tester.print_pdf()

    def plot_run_on_comparison():
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

    # plot_aspect_ratio()
    # plot_two_way_induction()
    # plot_run_on_comparison()
    # plot_run_on_convergence()
    # tabulate_run_on_convergence()

    tester.open_peter_converter()
    # tester.add_row(sample_files.joinpath(r"Aspect ratio\Maxwell"))
    # tester.close()
    app.exec_()
