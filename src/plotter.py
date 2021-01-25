import sys
import os
import pickle
import io
import re
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from natsort import natsorted, humansorted, os_sorted

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.pyplot import cm
from matplotlib.backends.backend_pdf import PdfPages

from src.file_types.fem_file import FEMFile, FEMTab
from src.file_types.tem_file import TEMFile, TEMTab
from src.file_types.platef_file import PlateFFile, PlateFTab
from src.file_types.mun_file import MUNFile, MUNTab
from PyQt5 import (QtCore, QtGui, uic)
from PyQt5.QtWidgets import (QMainWindow, QApplication, QMessageBox, QFrame, QErrorMessage, QFileDialog,
                             QTableWidgetItem, QScrollArea, QSpinBox, QHBoxLayout, QLabel, QInputDialog, QLineEdit,
                             QProgressDialog, QWidget, QHeaderView)

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
                component, ok_pressed = QInputDialog.getItem(self, "Choose Component", "Component:", components, 0, False)
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

        # Set the first column to stretch
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        self.remove_column = 7

        def change_pdf_path():
            filepath, ext = QFileDialog.getSaveFileName(self, 'Save PDF', '', "PDF Files (*.PDF)")
            self.pdf_filepath_edit.setText(str(Path(filepath).with_suffix(".PDF")))

        # Signals
        self.add_folder_btn.clicked.connect(self.add_row)
        self.change_pdf_path_btn.clicked.connect(change_pdf_path)
        self.table.cellClicked.connect(self.cell_clicked)

    def cell_clicked(self, row, col):
        print(f"Row {row}, column {col} clicked.")

        if col == self.remove_column:
            print(f"Removing row {row}.")
            self.table.removeRow(row)
            self.opened_files.pop(row)
            print(f"New opened files: {self.opened_files}")

    def add_row(self, folderpath=None):
        """Add a row to the table"""
        # File type options with extensions
        options = {"Maxwell": "*.TEM", "MUN": "*.DAT", "Peter": "*.XYZ", "PLATE": "*.DAT"}

        # Don't include filetypes that are already selected
        existing_filetypes = [self.table.item(row, 1).text() for row in range(self.table.rowCount())]
        for file_type in existing_filetypes:
            print(f"{file_type} already opened, removing from options.")
            del options[file_type]
            print(f"New options: {options}")

        # Don't add any  more rows if all file types have been selected
        if len(options) == 0:
            self.msg.information(self, "Maximum File Types Reached",
                                 "The maximum number of file types has been reached.")
            return

        if not folderpath:
            folderpath = QFileDialog.getExistingDirectory(self, "Select Folder", "")

        if Path(folderpath).is_dir():
            file_type, ok_pressed = QInputDialog.getItem(self, "Select File Type", "File Type:",
                                                         options.keys(), 0, False)

            if ok_pressed and file_type:
                row = self.table.rowCount()
                self.table.insertRow(row)

                ext = options[file_type]
                files = list(Path(folderpath).glob(ext))
                self.opened_files.append({file_type: files})

                # Create default items for each column
                path_item = QTableWidgetItem(folderpath)
                path_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                file_type_item = QTableWidgetItem(file_type)
                file_type_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                data_scaling = QTableWidgetItem("1.0")
                station_shift = QTableWidgetItem("0")
                start_ch = QTableWidgetItem("1")
                end_ch = QTableWidgetItem("99")
                files_found = QTableWidgetItem(str(len(files)))

                for col, item in enumerate([path_item, file_type_item, data_scaling, station_shift, start_ch, end_ch,
                                            files_found]):
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

                self.table.setCellWidget(row, self.remove_column, remove_btn_widget)

        else:
            self.msg.warning(self, "Error", f"{folderpath} does not exist.")
            return

    def print_pdf(self):
        """Create the PDF"""
        # Make sure the number of files found for each filetype is the same
        equal_num_files = all([self.table.item(row, 6).text() == self.table.item(0, 6).text()
                               for row in range(self.table.rowCount())])
        if equal_num_files is False:
            self.msg.critical(self, "Error", "Each file type must have equal number of files.")
            return

        t0 = time.time()

        files = zip()

        progress = QProgressDialog("Processing...", "Cancel", 0, len(files))
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setWindowTitle("Processing IRAP Files")


if __name__ == '__main__':
    import time
    import pandas as pd

    app = QApplication(sys.argv)

    # fpl = FEMPlotter()
    # fpl.show()
    # tpl = TEMPlotter()
    # tpl.show()

    sample_files = Path(__file__).parents[1].joinpath('sample_files')

    def auto_run_files():
        t0 = time.time()
        maxwell_files_folder = sample_files.joinpath(r"Aspect ratio test\Maxwell")
        plate_files_folder = sample_files.joinpath(r"Aspect ratio test\PLATE")

        maxwell_files = os_sorted(maxwell_files_folder.glob("*.TEM"))
        plate_files = os_sorted(plate_files_folder.glob("*.DAT"))

        assert len(maxwell_files) == len(plate_files), \
            print(f"{len(maxwell_files)} Maxwell files vs {len(plate_files)} PLATE files found.")

        results_pdf = sample_files.joinpath(r"Aspect ratio test/Aspect Ratio Test.pdf")

        files = list(zip(maxwell_files, plate_files))[:]
        filepath, ext = QFileDialog.getSaveFileName(None, 'Save PDF', str(results_pdf),
                                                    "PDF Files (*.PDF);;All Files (*.*)")

        with PdfPages(filepath) as pdf:
            progress = QProgressDialog("Processing...", "Cancel", 0, len(files))
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setWindowTitle("Processing IRAP Files")

            for ind, (maxwell_file, plate_file) in enumerate(files):
                if progress.wasCanceled():
                    break
                progress.setValue(ind)
                print(f"Plotting files: {maxwell_file.name}, {plate_file.name} ({ind + 1}/{len(files)})")
                tpl.open(maxwell_file)
                tpl.open(plate_file)

                # df = pd.DataFrame(zip(tpl.file_tab_widget.widget(0).widget().file.ch_times[20:44], tpl.file_tab_widget.widget(
                #     1).widget().file.ch_times * 1000), columns=['Maxwell', 'PLATE'], dtype=float)
                #
                # df['Difference'] = df.Maxwell - df.PLATE

                tpl.title.setText("Aspect Ratio Test")
                tpl.title.editingFinished.emit()
                tpl.file_tab_widget.widget(0).widget().scale_data_sbox.setValue(0.000001)
                tpl.file_tab_widget.widget(0).widget().shift_stations_sbox.setValue(-400)
                tpl.file_tab_widget.widget(0).widget().min_ch.setValue(21)
                tpl.file_tab_widget.widget(0).widget().max_ch.setValue(44)

                tpl.file_tab_widget.widget(0).widget().alpha_sbox.setValue(50)

                # Print every figure as a PDF page
                for figure in [tpl.x_figure, tpl.y_figure, tpl.z_figure]:

                    # Only print the figure if there are plotted lines
                    if figure.axes[0].lines:
                        old_size = figure.get_size_inches().copy()
                        figure.set_size_inches((11, 8.5))
                        # figure.axes[0].set_xlim([0, 200])
                        pdf.savefig(figure, orientation='landscape')
                        figure.set_size_inches(old_size)

                tpl.remove_tab(0)
                tpl.remove_tab(0)

        print(f"Script complete after {(time.time() - t0) / 60:.0f}min {(time.time() - t0) % 60:.0f}s.")
        os.startfile(filepath)

    # tem_file = sample_files.joinpath(r'MUN files\LONG_V1x1_450_50_100_50msec_3D_solution_channels_tem_time_decay_z.dat')
    # tem_file = sample_files.joinpath(r'MUN files\LONG_V1x1_450_50_100_50msec_3D_solution_channels_tem_time_decay_y.dat')
    # tem_file = sample_files.joinpath(r'PLATEF files\450_50.dat')
    fem_file = sample_files.joinpath(r'Maxwell files\FEM\Horizontal Plate 100S Normalized.fem')
    # fem_file = sample_files.joinpath(r'Maxwell files\FEM\Test 4 FEM files\Test 4 - h=5m.fem')
    # fem_file = sample_files.joinpath(r'Maxwell files\FEM\Turam 2x4 608S_0.96691A_PFCALC at 1A.fem')
    # fem_file = sample_files.joinpath(r'Maxwell files\FEM\test Z.fem')
    # tem_file = sample_files.joinpath(r'Maxwell files\TEM\V_1x1_450_50_100 50msec instant on-time first.tem')
    # tem_file = sample_files.joinpath(r'Maxwell files\TEM\50msec Impulse 100S BField.tem')
    # tem_file = sample_files.joinpath(r'Maxwell files\TEM\Test 6 - x1e3.tem')
    tem_file = sample_files.joinpath(r'Aspect ratio test\Maxwell\5x150A.TEM')

    # fpl.show()
    # fpl.open(fem_file)

    # tpl.show()
    # tpl.open(tem_file)
    # tpl.print_pdf()

    # auto_run_files()
    tester = TestRunner()
    tester.show()
    # tester.pdf_filepath_edit.setText(r"C:\Users\Mortulo\PycharmProjects\IRAP_Modelling\sample_files\Aspect ratio test\test.pdf")
    # tester.add_row(folderpath=r"C:\Users\Mortulo\PycharmProjects\IRAP_Modelling\sample_files\Aspect ratio test\Maxwell")
    # tester.add_row(folderpath=r"C:\Users\Mortulo\PycharmProjects\IRAP_Modelling\sample_files\Aspect ratio test\PLATE")
    # tester.print_pdf()

    app.exec_()
