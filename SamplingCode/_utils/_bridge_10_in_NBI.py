# -----------------------------------------------------------------------------------#
# To reporoduce the example in the GIAMS paper
# -----------------------------------------------------------------------------------#
from ._encode import encode_bridge_parameters

def get_bridge_10_NBI():

	prm = {'id' : hash(str(np.random.random()*np.random.random())), 
			'length' : (54.3 - 5) / 1795,
			'width' : (16.8 - 3) / 57,
			'vertical_clearance': (7-4) / 3,
			'ADT': (12797 - 100) / 399900,
			'truck_percentage': 0.05 / 0.5,
			'detour_length': (6 - 1) / 99,
			'skew_angle': 6 / 45,
			'n_spans': (3 - 1) / 59,
			'maint_duration': (30 - 10) / 50,
			'rehab_duration': (180 - 120) / 120,
			'recon_duration': (360 - 300) / 240,
			'speed_before': (60 - 40) / 50,
			'speed_after': (30 - 15) / 20,
			'drift': (0.1 - 0.01) / 0.09,
			'volatility': (0.01 - 0.01) / 0.09,
			'detour_usage_percentage': 0.1 / 0.99,
			'occurrence_rate': (0.3 - 0.001) / 0.099,
			'dist_first_param': (2.1739 - 3) / 2,
			'dist_second_param': (4 - 0.01) / 1.99,
			'deck_cond': (5 - 4) / 5,
			'deck_age': (14 - 1) / 89,
			'superstructure_cond': (7 - 4) / 5,
			'superstructure_age': (14 - 1) / 89,
			'substructure_cond': (7 - 4) / 5,
			'substructure_age': (14 - 1) / 89
			}

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
		chosen_one = np.random.choice(ls)
		for i, item in enumerate(ls[1:]):
			prm[f'{key}_{item}'] = 0

	prm['material_4'] = 1
	prm['design_2'] = 1
	prm['hazus_class_HWB15'] = 1
	prm['road_class_Major'] = 1
	prm['site_class_B'] = 1
	## Since the deck material is 1, we don't set it as 1
	## prm['deck_material'] = 1
	return prm