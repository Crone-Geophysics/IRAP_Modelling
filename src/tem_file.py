import pandas as pd


class TEMFile:

    def __init__(self):

        self.filepath = None
        self.config = 'fixed_loop'
        self.elevation = 0.
        self.units = 'nT/s'
        self.current = 1.
        self.tx_turns = 1.
        self.base_freq = 1.
        self.duty_cycle = 50.
        self.on_time = 50.
        self.off_time = 50.
        self.turn_on = 0.
        self.turn_off = 0.
        self.timing_mark = 0.
        self.rx_area_x = 4000.
        self.rx_area_y = 4000.
        self.rx_area_z = 4000.
        self.rx_dipole = False
        self.tx_dipole = False
        self.loop = None
        self.loop = pd.DataFrame(columns=['Easting', 'Northing', 'Elevation'], dtype=float)
        self.ch_times = []
        self.ch_widths = []
        self.data = pd.DataFrame()


"""
TEM file object
"""