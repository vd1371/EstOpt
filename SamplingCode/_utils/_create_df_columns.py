def create_df_columns(gaopt_instance):

	cols = list(gaopt_instance.create_bridge().keys())[1:]
	# Three elements were used in the original project
	for elem in range(gaopt_instance.n_elements):
		# 20 years, biennially, binary representation --> 20 variables
		for step in range(gaopt_instance.dt*gaopt_instance.n_steps):
			cols.append(f'Eelem{elem}-{step}')
	cols.append("obj")

	return cols

