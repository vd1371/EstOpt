import os
import pandas as pd

def _load_and_concat_all_csvs(direc, N = -1):

	holder = []
	files = os.listdir(direc) if N is None else os.listdir(direc)[:N]

	for file_name in files:
		df = pd.read_csv(direc + "/" + file_name, index_col = 0)
		holder.append(df)

	df = pd.concat(holder)

	return df