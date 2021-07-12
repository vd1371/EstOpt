import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def get_classification_report(data, model, step, ne, logger):

	y_true = data.loc[:, f"Elem{ne}-{step}"]

	X = data.loc[:, "length": "deck_material_8"]

	if model == 0:
		y_pred = np.random.choice([0, 1, 2, 3], size = (len(y_true)))
	else:
		y_pred = model.predict(X)

	logger.info(f"----------Classification Report for- Step:{step}-ne:{ne}------------\n" + \
						str(classification_report(y_true, y_pred))+"\n")
	logger.info(f"----------Confusion Matrix for - Step:{step}-ne:{ne}------------\n" + \
						str(confusion_matrix(y_true, y_pred))+"\n")
	logger.info(f'----------Accurcay for {step}-{ne}------------\n' + \
						str(round(accuracy_score(y_true, y_pred),4)))

	output_dict = {'accuracy': accuracy_score(y_true, y_pred),
					'precision': precision_score(y_true, y_pred, average='macro'),
					'recall': recall_score(y_true, y_pred, average='macro'),
					'f1_score': f1_score(y_true, y_pred, average='macro')}

	return output_dict




