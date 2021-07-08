import numpy as np
import pprint
from ._encode import encode_bridge_parameters

def create_synthesized_assets():

	prm = {'id' : hash(str(np.random.random()*np.random.random())), 
			'length' : np.random.randint(5, 1800),
			'width' : np.random.randint(3, 60),
			'vertical_clearance': np.random.uniform(4, 7),
			'ADT': np.random.randint(100, 400000),
			'truck_percentage': np.random.uniform(0, 0.5),
			'detour_length': np.random.randint(1, 100),
			'skew_angle': np.random.randint(0, 45),
			'n_spans': int(np.random.randint(1, 60)),
			'maint_duration': np.random.randint(10, 60),
			'rehab_duration': np.random.randint(120, 240),
			'recon_duration': np.random.randint(300, 540),
			'speed_before': np.random.randint(40, 90),
			'speed_after': np.random.randint(15, 35),
			'drift': np.random.uniform(0.01, 0.1),
			'volatility': np.random.uniform(0.01, 0.1),
			'detour_usage_percentage': np.random.uniform(0, 0.99),
			'occurrence_rate': np.random.uniform(0.001, 0.1),
			'dist_first_param': np.random.uniform(3, 5),
			'dist_second_param': np.random.uniform(0.01, 2),
			'deck_cond': np.random.choice([9, 8, 7, 6, 5, 4]),
			'deck_age': np.random.randint(1, 90),
			'superstructure_cond': np.random.choice([9, 8, 7, 6, 5, 4]),
			'superstructure_age': np.random.randint(1, 90),
			'substructure_cond': np.random.choice([9, 8, 7, 6, 5, 4]),
			'substructure_age': np.random.randint(1, 90),
			'material': np.random.choice([1, 2, 3, 4, 5, 6]),
			'design': np.random.choice([1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 14, 16, 21, 22]),
			'hazus_class': 'HWB' + str(np.random.choice([1, 10, 12, 15, 17, 22, 3, 5, 8])),
			'road_class': np.random.choice(['Local', 'Major', 'Minor', 'NHS']),
			'site_class' : np.random.choice(['A', 'B', 'C']),
			'deck_material' : np.random.choice([1, 2, 3, 8]),
				}

	return encode_bridge_parameters(prm)

if __name__ == "__main__":
	create_synthesized_assets()