import sys
import os
import pickle
import io
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from natsort import natsorted

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.pyplot import cm
from matplotlib.backends.backend_pdf import PdfPages

from src.file_types.fem_file import FEMFile, FEMTab
from src.file_types.tem_file import TEMFile, TEMTab
from src.file_types.platef_file import PlateFFile, PlateFTab
from src.file_types.mun_file import MUNFile, MUNTab
from PyQt5 import (QtCore, uic)
from PyQt5.QtWidgets import (QMainWindow, QApplication, QMessageBox, QFrame, QErrorMessage, QFileDialog,
                             QScrollArea, QSpinBox, QHBoxLayout, QLabel, QGroupBox, QCheckBox, QButtonGroup)

# Modify the paths for when the script is being run in a frozen state (i.e. as an EXE)
if getattr(sys, 'frozen', False):
    application_path = Path(sys.executable).parent
    FEMPlotterUIFile = Path('ui\\fem_plotter.ui')
    TEMPlotterUIFile = Path('ui\\tem_plotter.ui')
    icons_path = Path('ui\\icons')
else:
    application_path = Path(__file__).absolute().parent
    FEMPlotterUIFile = application_path.joinpath('ui\\fem_plotter.ui')
    TEMPlotterUIFile = application_path.joinpath('ui\\tem_plotter.ui')
    icons_path = application_path.joinpath('ui\\icons')

# Load Qt ui file into a class
fem_plotterUI, _ = uic.loadUiType(FEMPlotterUIFile)
tem_plotterUI, _ = uic.loadUiType(TEMPlotterUIFile)

matplotlib.use('Qt5Agg')
rainbow_colors = iter(cm.rainbow(np.linspace(0, 1, 20)))
quant_colors = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
# color = iter(cm.tab10())


class FEMPlotter(QMainWindow, fem_plotterUI):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setAcceptDrops(True)
        self.setWindowTitle("IRAP FEM Plotter")
        self.resize(800, 600)
        # self.setWindowIcon(QtGui.QIcon(str(icons_path.joinpath('voltmeter.png'))))
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

        # Status bar
        self.alpha_frame = QFrame()
        self.alpha_frame.setLayout(QHBoxLayout())
        self.alpha_frame.layout().addWidget(QLabel("Plot Alpha: "))
        self.alpha_frame.setMaximumWidth(150)

        self.alpha_sbox = QSpinBox()
        self.alpha_sbox.setSingleStep(10)
        self.alpha_sbox.setRange(0, 100)
        self.alpha_sbox.setValue(100)

        self.alpha_frame.layout().addWidget(self.alpha_sbox)
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
            for canvas, ax in zip([self.hcp_canvas, self.vca_canvas], [self.hcp_ax, self.vca_ax]):
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
            color = next(quant_colors)  # Cycles through colors
        except StopIteration:
            self.msg.showMessage(self, "Error", "Maximum number of files reached.")
            return

        # Create a dict for which axes components get plotted on
        axes = {'HCP': self.hcp_ax, 'VCA': self.vca_ax}

        # Create a new tab and add it to the widget
        if ext == '.fem':
            tab = FEMTab(parent=self, color=color, axes=axes)
        else:
            tab = None

        # try:
        tab.read(filepath)
        # except Exception as e:
        #     self.err_msg.showMessage(str(e))
        #     return

        # Connect signals
        tab.toggle_sig.connect(self.update_legend)

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

        for canvas, ax in zip([self.hcp_canvas, self.vca_canvas], [self.hcp_ax, self.vca_ax]):
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
        tab.remove()
        self.opened_files.pop(ind)
        self.file_tab_widget.removeTab(ind)

        self.update_legend()
        self.update_num_files()

    def update_legend(self):
        """Update the legend to be in alphabetical order"""

        for canvas, ax in zip([self.hcp_canvas, self.vca_canvas], [self.hcp_ax, self.vca_ax]):
            if self.actionPlot_Legend.isChecked():
                # Only sort if there are tabs, otherwise it crashes.
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    # sort both labels and handles by labels
                    labels, handles = zip(*natsorted(zip(labels, handles), key=lambda t: t[0]))
                    ax.legend(handles, labels)
                else:
                    ax.legend()
            else:
                legend = ax.get_legend()
                if legend:
                    legend.remove()

            canvas.draw()
            canvas.flush_events()

    def update_alpha(self, alpha):
        print(f"New alpha: {alpha / 100}")
        for canvas, ax in zip([self.hcp_canvas, self.vca_canvas], [self.hcp_ax, self.vca_ax]):

            for artist in ax.lines:
                artist.set_alpha(alpha / 100)

            for artist in ax.collections:
                artist.set_alpha(alpha / 100)

            canvas.draw()
            canvas.flush_events()

        self.update_legend()

    def update_num_files(self):
        self.statusBar().showMessage(f"{len(self.opened_files)} file(s) opened.")


class TEMPlotter(QMainWindow, tem_plotterUI):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setAcceptDrops(True)
        self.setWindowTitle("IRAP TEM Plotter")
        self.resize(800, 600)
        # self.setWindowIcon(QtGui.QIcon(str(icons_path.joinpath('voltmeter.png'))))
        self.err_msg = QErrorMessage()
        self.msg = QMessageBox()
        self.opened_files = []

        # X Figure
        self.x_figure = Figure()
        self.x_ax = self.x_figure.add_subplot(111)
        self.x_ax.set_title('X Component')
        self.x_ax.set_xlabel("Station")
        self.x_canvas = FigureCanvas(self.x_figure)

        toolbar = NavigationToolbar(self.x_canvas, self)

        self.x_canvas_frame.layout().addWidget(self.x_canvas)
        self.x_canvas_frame.layout().addWidget(toolbar)

        # Y Figure
        self.y_figure = Figure()
        self.y_ax = self.y_figure.add_subplot(111)
        self.y_ax.set_title('Y Component')
        self.y_ax.set_xlabel("Station")
        self.y_canvas = FigureCanvas(self.y_figure)

        toolbar = NavigationToolbar(self.y_canvas, self)

        self.y_canvas_frame.layout().addWidget(self.y_canvas)
        self.y_canvas_frame.layout().addWidget(toolbar)

        # Z Figure
        self.z_figure = Figure()
        self.z_ax = self.z_figure.add_subplot(111)
        self.z_ax.set_title('Z Component')
        self.z_ax.set_xlabel("Station")
        self.z_canvas = FigureCanvas(self.z_figure)

        toolbar = NavigationToolbar(self.z_canvas, self)

        self.z_canvas_frame.layout().addWidget(self.z_canvas)
        self.z_canvas_frame.layout().addWidget(toolbar)

        # Status bar
        self.alpha_frame = QFrame()
        self.alpha_frame.setLayout(QHBoxLayout())
        self.alpha_frame.layout().addWidget(QLabel("Plot Alpha: "))
        self.alpha_frame.setMaximumWidth(150)

        self.alpha_sbox = QSpinBox()
        self.alpha_sbox.setSingleStep(10)
        self.alpha_sbox.setRange(0, 100)
        self.alpha_sbox.setValue(100)

        self.alpha_frame.layout().addWidget(self.alpha_sbox)

        # Legend option
        self.legend_box = QFrame()
        self.legend_box.setLayout(QHBoxLayout())
        self.color_by_line_cbox = QCheckBox("Color by Line")
        self.color_by_channel_cbox = QCheckBox("Color by Channel")
        self.legend_color_group = QButtonGroup()
        self.legend_color_group.addButton(self.color_by_line_cbox)
        self.legend_color_group.addButton(self.color_by_channel_cbox)
        self.legend_box.layout().addWidget(self.color_by_line_cbox)
        self.legend_box.layout().addWidget(self.color_by_channel_cbox)
        self.color_by_line_cbox.setChecked(True)

        self.statusBar().addPermanentWidget(self.legend_box)
        self.statusBar().addPermanentWidget(self.alpha_frame)

        # Signals
        self.actionOpen.triggered.connect(self.open_file_dialog)
        self.actionPrint_to_PDF.triggered.connect(self.print_pdf)
        self.actionPlot_Legend.triggered.connect(self.update_legend)
        self.legend_color_group.buttonClicked.connect(lambda: self.plot_tab(self.file_tab_widget.currentIndex()))

        self.file_tab_widget.tabCloseRequested.connect(self.remove_tab)
        self.alpha_sbox.valueChanged.connect(self.update_alpha)

        self.update_num_files()

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Space:
            for ax, canvas in zip([self.x_ax, self.y_ax, self.z_ax], [self.x_canvas, self.y_canvas, self.z_canvas]):
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

    def print_pdf(self):
        """Resize the figure to 11 x 8.5" and save to a PDF file"""

        if not any([self.x_ax.lines, self.y_ax.lines, self.z_ax.lines]):
            self.statusBar().showMessage(f"The plots are empty.", 1500)
            print(f"The plots are empty.")
            return

        filepath, ext = QFileDialog.getSaveFileName(self, 'Save PDF', '', "PDF Files (*.PDF);;All Files (*.*)")

        if filepath:
            with PdfPages(filepath) as pdf:
                # Print every figure as a PDF page
                for figure in [self.x_figure, self.y_figure, self.z_figure]:

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

        try:
            color = next(quant_colors)  # Cycles through colors
        except StopIteration:
            self.msg.showMessage(self, "Error", "Maximum number of files reached.")
            return

        # Create a dict for which axes components get plotted on
        axes = {'X': self.x_ax, 'Y': self.y_ax, 'Z': self.z_ax}

        # Create a new tab and add it to the widget
        if ext == '.tem':
            tab = TEMTab(parent=self, color=color, axes=axes)
        elif ext == '.dat':
            first_line = open(filepath).readlines()[0]
            if 'Data type:' in first_line:
                tab = MUNTab(parent=self, color=color, axes=axes)
            else:
                tab = PlateFTab(parent=self, color=color, axes=axes)
        else:
            tab = None

        # try:
        tab.read(filepath)
        # except Exception as e:
        #     self.err_msg.showMessage(str(e))
        #     return

        # Connect signals
        tab.toggle_sig.connect(self.update_legend)  # Update the legend when the plot is toggled

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

        tab.plot(alpha, color_by_channel=self.color_by_channel_cbox.isChecked())

        for canvas, ax in zip([self.x_canvas, self.y_canvas, self.z_canvas], [self.x_ax, self.y_ax, self.z_ax]):
            # Add the Y axis label
            if not ax.get_ylabel() or self.file_tab_widget.count() == 1:
                ax.set_ylabel(tab.file.units)
            else:
                if ax.get_ylabel() != tab.file.units:
                    print(f"Warning: The units for {tab.file.filepath.name} are different then the prior units.")
                #     self.msg.warning(self, "Warning", f"The units for {tab.file.filepath.name} are"
                #                                       f" different then the prior units.")

            # Update the plot
            canvas.draw()
            canvas.flush_events()

        self.update_legend()

    def remove_tab(self, ind):
        """Remove a tab"""
        # Find the tab when an index is passed (when a tab is closed)
        tab = self.file_tab_widget.widget(ind).widget()
        tab.remove()
        self.opened_files.pop(ind)
        self.file_tab_widget.removeTab(ind)

        self.update_legend()
        self.update_num_files()

    def update_legend(self):
        """Update the legend to be in alphabetical order"""

        for canvas, ax in zip([self.x_canvas, self.y_canvas, self.z_canvas], [self.x_ax, self.y_ax, self.z_ax]):
            if self.actionPlot_Legend.isChecked():
                # Only sort if there are tabs, otherwise it crashes.
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    # sort both labels and handles by labels
                    labels, handles = zip(*natsorted(zip(labels, handles), key=lambda t: t[0]))
                    ax.legend(handles, labels)
                else:
                    ax.legend()
            else:
                legend = ax.get_legend()
                if legend:
                    legend.remove()

            canvas.draw()
            canvas.flush_events()

    def update_alpha(self, alpha):
        print(f"New alpha: {alpha / 100}")
        for canvas, ax in zip([self.x_canvas, self.y_canvas, self.z_canvas], [self.x_ax, self.y_ax, self.z_ax]):

            for artist in ax.lines:
                artist.set_alpha(alpha / 100)

            for artist in ax.collections:
                artist.set_alpha(alpha / 100)

            canvas.draw()
            canvas.flush_events()

        self.update_legend()

    def update_num_files(self):
        self.statusBar().showMessage(f"{len(self.opened_files)} file(s) opened.")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    sample_files = Path(__file__).parents[1].joinpath('sample_files')

    # mun_file = sample_files.joinpath(r'MUN files\LONG_V1x1_450_50_100_50msec_3D_solution_channels_tem_time_decay_z.dat')
    # platef_file = sample_files.joinpath(r'PLATEF files\450_50.dat')
    # fem_file = sample_files.joinpath(r'Maxwell files\FEM\Horizontal Plate 100S Normalized.fem')
    # fem_file = sample_files.joinpath(r'Maxwell files\FEM\Test 4 FEM files\Test 4 - h=5m.fem')
    fem_file = sample_files.joinpath(r'Maxwell files\FEM\Turam 2x4 608S_0.96691A_PFCALC at 1A.fem')
    # fem_file = sample_files.joinpath(r'Maxwell files\FEM\test Z.fem')
    tem_file = sample_files.joinpath(r'Maxwell files\TEM\V_1x1_450_50_100 50msec instant on-time first.tem')
    # tem_file = sample_files.joinpath(r'Maxwell files\TEM\50msec Impulse 100S BField.tem')

    fpl = FEMPlotter()
    fpl.show()
    fpl.open(fem_file)

    # tpl = TEMPlotter()
    # tpl.show()
    # tpl.open(tem_file)

    app.exec_()
