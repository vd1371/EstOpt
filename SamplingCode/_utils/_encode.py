import numpy as np

def encode_bridge_parameters(prm):

	new_bridge_params = {}

	new_bridge_params['id'] = prm['id']
	new_bridge_params['length'] = (prm['length']-5) / 1795
	new_bridge_params['width'] = (prm['width'] - 3) / 57
	new_bridge_params['vertical_clearance'] = (prm['vertical_clearance'] - 4) /3
	new_bridge_params['ADT'] = (prm['ADT'] - 100) / 399900
	new_bridge_params['truck_percentage'] = prm['truck_percentage'] / 0.5
	new_bridge_params['detour_length'] = (prm['detour_length'] - 1) / 99
	new_bridge_params['skew_angle'] = prm['skew_angle'] / 45
	new_bridge_params['n_spans'] = (prm['n_spans'] - 1) / 59
	new_bridge_params['maint_duration'] = (prm['maint_duration'] - 10) / 50
	new_bridge_params['rehab_duration'] = (prm['rehab_duration'] - 120) / 120
	new_bridge_params['recon_duration'] = (prm['recon_duration'] - 300) / 240
	new_bridge_params['speed_before'] = (prm['speed_before'] - 40) / 50
	new_bridge_params['speed_after'] = (prm['speed_after'] - 15) / 20
	new_bridge_params['drift'] = (prm['drift'] - 0.01) / 0.09
	new_bridge_params['volatility'] = (prm['volatility'] - 0.01) / 0.09
	new_bridge_params['detour_usage_percentage'] = prm['detour_usage_percentage'] / 0.99
	new_bridge_params['occurrence_rate'] = (prm['occurrence_rate'] - 0.001) / 0.099
	new_bridge_params['dist_first_param'] = (prm['dist_first_param'] - 3) / 2
	new_bridge_params['dist_second_param'] = (prm['dist_second_param'] - 0.01) / 1.99
	new_bridge_params['deck_cond'] = (prm['deck_cond'] - 4) / 5
	new_bridge_params['deck_age'] = (prm['deck_age'] - 1) / 89
	new_bridge_params['superstructure_cond'] = (prm['superstructure_cond'] - 4) / 5
	new_bridge_params['superstructure_age'] = (prm['superstructure_age'] - 1) / 89
	new_bridge_params['substructure_cond'] = (prm['substructure_cond'] - 4) / 5
	new_bridge_params['substructure_age'] = (prm['substructure_age'] - 1) / 89

	categorical_info_dict = {
		'material': [1, 2, 3, 4, 5, 6],
		'design' : [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 14, 16, 21, 22],
		'hazus_class' : [f'HWB{val}' for val in [1, 10, 12, 15, 17, 22, 3, 5, 8,]],
		'road_class' : ['Local', 'Major', 'Minor', 'NHS'],
		'site_class' : ['A', 'B', 'C'],
		'deck_material' : [1, 2, 3, 8]
		}

	# Adding encoded parameters
	for key, ls in categorical_info_dict.items():
		for i, item in enumerate(ls[1:]):
			new_bridge_params[f'{key}_{item}'] = 1 if item == prm[key] else 0

	return new_bridge_params