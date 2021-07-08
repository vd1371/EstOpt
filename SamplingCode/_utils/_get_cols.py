def get_cols(df, ls):
	output = []
	for item in ls:
		for col in df.columns:
			if item in col: output.append(col)
	return output