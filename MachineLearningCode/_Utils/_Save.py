import joblib

def save_model(model, direc, name):
	joblib.dump(model, f"{direc}/{name}.jbl.lzma" , compress=3)
