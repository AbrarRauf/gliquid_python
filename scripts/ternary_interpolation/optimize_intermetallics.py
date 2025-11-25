from ternary_HSX import ternary_gtx_plotter
import plotly.offline as ploff
from gliquid.config import data_dir
import os
import json
import pandas as pd

dump_dir = "all_dumps/optimize/"
read_dir = "all_dumps/binary_fits/"

if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

def compute_ternary(tern_sys):
    os.environ["NEW_MP_API_KEY"] = "Rtb4ppAs9rcNVzh10IVdBRh6HwlBymcJ"
    tern_param_format = 'combined'

    binary_param_df = pd.read_excel("data/ternary_dft_data/multi_fit_no1S_nmae_lt_0.5.xlsx")

