import numpy as np
import pandas as pd
import os

from ._get_direc_to_data import _get_direc_to_data
from ._load_and_concat_all_csvs import _load_and_concat_all_csvs
from ._covert_binary_plans_to_decimal import _covert_binary_plans_to_decimal

def load_indiana_data():

	direc = _get_direc_to_data()
	df = _load_and_concat_all_csvs(direc, N = None)
	df = _covert_binary_plans_to_decimal(df)

	return df