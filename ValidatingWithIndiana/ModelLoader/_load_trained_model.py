import joblib

from ._get_direc_to_models import _get_direc_to_models

def load_trained_model(step, ne):

	direc = _get_direc_to_models()

	print (f"Trying to load model for element {ne} at step {step}...")

	# return 0

	model_number = ne * 10 + step

	if step % 2 == 0:
		model = joblib.load(direc + "/" + f"RF-{model_number}.jbl.lzma")
	else:
		model = joblib.load(direc + "/" + f"RF-1.jbl.lzma")

	print ("Loaded.")
	return model