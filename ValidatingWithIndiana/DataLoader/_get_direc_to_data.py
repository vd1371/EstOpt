import os

def _get_direc_to_data():

	direc = rf"{os.path.dirname(__file__)}"
	direc = direc.split(os.sep)
	direc[0] = direc[0] + os.sep

	direc_to_root = os.path.join(*direc[:-3])
	direc_to_data = os.path.join(direc_to_root, 'Data', "ValidationWithIndiana/")

	return direc_to_data