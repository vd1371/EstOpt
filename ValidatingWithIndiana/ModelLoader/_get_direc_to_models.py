import os

def _get_direc_to_models():

	direc = rf"{os.path.dirname(__file__)}"
	direc = direc.split(os.sep)
	direc[0] = direc[0] + os.sep

	direc_to_root = os.path.join(*direc[:-2])
	direc_to_data = os.path.join(direc_to_root, "MachineLearningCode", "Reports", "RF/")

	return direc_to_data