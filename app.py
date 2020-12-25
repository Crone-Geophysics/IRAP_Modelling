import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
from typing import Dict
from src.file_types.fem_file import FEMFile
from src.file_types.tem_file import TEMFile
from src.file_types.platef_file import PlateFFile
from src.file_types.mun_file import MUNFile
from pathlib import Path


@st.cache(allow_output_mutation=True)
def get_static_store() -> Dict:
    """This dictionary is initialized once and can be used to store the files uploaded"""
    return {}


def read_file(file):
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
        # self.msg.showMessage(self, 'Error', f"{ext[1:]} is not an implemented file extension.")
        f = None

    return f


def main():
    static_store = get_static_store()
    st.subheader("IRAP Plotter")

    data = st.file_uploader("Upload a Dataset", type=["fem", "tem", "dat"])
    if data:
        value = data.getvalue()
        # And add it to the static_store if not already in
        if value not in static_store.values():
            static_store[data] = value

            file = data.read()
            print(file)
    else:
        static_store.clear()  # Hack to clear list if the user clears the cache and reloads the page
        st.info("Upload one or more `.py` files.")


main()

