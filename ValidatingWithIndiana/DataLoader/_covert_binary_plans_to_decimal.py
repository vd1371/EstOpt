import pandas as pd
import numpy as np

def _covert_binary_plans_to_decimal(df):

	df = df.drop(columns  = ['obj'])

	idx = list(df.columns).index('Eelem0-0')

	n_elements = 3
	n_steps = 10
	dt = 2

	for ne in range(n_elements):
		for step in range(n_steps):

			df[f"Elem{ne}-{step}"] = (2 * df[f"Eelem{ne}-{step*dt}"] + \
									df[f"Eelem{ne}-{step*dt + 1}"]).astype('int')

	columns_to_drop = []
	for col in df.columns:
		if "Eelem" in col:
			columns_to_drop.append(col)
	df = df.drop(columns  = columns_to_drop)

	return df