import numpy as np
from ._get_cols import get_cols

def eval_gener(df, check_policy_obj, **params):

		user_cost_model = params.pop("user_cost_model")
		agency_cost_model = params.pop("agency_cost_model")
		utility_model = params.pop("utility_model")
		sorting_order = params.pop("sorting_order")
		

		user_costs = user_cost_model(
						df.drop(
							columns = get_cols(df, ['UserCost', 'AgencyCost',
													'Utility', 'Obj',
													'width', 'vertical_clearance',
													'design_'])
							), 
						training = False).numpy()

		# user_costs = user_cost_model.predict(
		# 				df.drop(
		# 					columns = get_cols(df, ['UserCost', 'AgencyCost',
		# 											'Utility', 'Obj',
		# 											'width', 'vertical_clearance',
		# 											'design_'])
		# 					))

		# agency_costs = agency_cost_model(
		# 				df.drop(
		# 					columns = get_cols(df, ['UserCost', 'AgencyCost',
		# 											'Utility', 'Obj',
		# 											'ADT', 'truck_percentage',
		# 											'detour_length',
		# 											'_duration', 'speed_',
		# 											'drift', 'volatility',
		# 											'detour_usage_percentage'])
		# 					),
		# 				training = False).numpy()

		utilities = utility_model(
						df.drop(
							columns = get_cols(df, ['UserCost', 'AgencyCost',
													'Utility', 'Obj',
													'length', 'width',
													'vertical_clearance',
													'design_', 'ADT',
													'truck_percentage',
													'_duration',
													'speed_',
													'drift',
													'volatility',
													'detour_usage_percentage'])
								),
						training = False).numpy()

		# utilities = utility_model.predict(
		# 				df.drop(
		# 					columns = get_cols(df, ['UserCost', 'AgencyCost',
		# 											'Utility', 'Obj',
		# 											'length', 'width',
		# 											'vertical_clearance',
		# 											'design_', 'ADT',
		# 											'truck_percentage',
		# 											'_duration',
		# 											'speed_',
		# 											'drift',
		# 											'volatility',
		# 											'detour_usage_percentage'])
		# 						))


		# Finding the objective function (It is currently based on GIAMS example 1)
		user_costs = user_costs.reshape(-1)
		user_costs[user_costs < 0] = np.inf
		user_costs = user_costs.reshape(-1, 1)

		df['Obj'] = (utilities / user_costs** 0.2)

		# Checking the values
		df['Obj'] = df.loc[:, 'Eelem0-0': 'Obj'].apply(check_policy_obj, axis = 1)

		# Sorting values
		df.sort_values('Obj', inplace = True, ascending = sorting_order)

		return df

