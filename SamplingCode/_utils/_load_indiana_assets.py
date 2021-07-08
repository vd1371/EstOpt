import os
import pandas as pd

from ._encode import encode_bridge_parameters

def load_indiana_assets(N = None):

	script_path = os.path.dirname(__file__)
	df = pd.read_csv(script_path + "/INDIANA/INDIANA2019.csv", index_col = 0)

	if not N is None:
		df = df.iloc[:N, :]

	all_assets = {}

	for index, row in df.iterrows():
		prm = {'id' : row['STRUCTURE_NUMBER_008'], 
			'length' : row['STRUCTURE_LEN_MT_049'],
			'width' : row['DECK_WIDTH_MT_052'],
			'material': row['STRUCTURE_KIND_043A'],
			'design': row['STRUCTURE_TYPE_043B'],
			'road_class': row['Road_class'],
			'ADT': row['FUTURE_ADT_114'],
			'truck_percentage': row['PERCENT_ADT_TRUCK_109'],
			'detour_length': row['DETOUR_KILOS_019'],
			'hazus_class': row['HAZUS_classficiation'],
			'site_class': row['site_class'],
			'skew_angle': row['DEGREES_SKEW_034'],
			'n_spans': row['MAIN_UNIT_SPANS_045'],
			'maint_duration': row['maint_duration'],
			'rehab_duration': row['rehab_duration'],
			'recon_duration': row['recon_duration'],
			'deck_age': row['Age'],
			'superstructure_age': row['Age'],
			'substructure_age': row['Age'],
			'deck_material' : row['DECK_STRUCTURE_TYPE_107'],
			'deck_cond': row['DECK_COND_058'],
			'superstructure_cond': row['SUPERSTRUCTURE_COND_059'],
			'substructure_cond': row['SUBSTRUCTURE_COND_060'],
			'vertical_clearance': min(row['Vertical_clearance'], 7),
			'speed_before': 60,
			'speed_after': 30,
			'drift': 0.1,
			'volatility': 0.01,
			'detour_usage_percentage': 0.1,
			'occurrence_rate': 0.1,
			'dist_first_param': 3,
			'dist_second_param': 2}

		all_assets[index] = encode_bridge_parameters(prm)

	return all_assets




