import pandas as pd
import time

def save_batch(direc, holder, cols, indices, batch_number, start):

	df = pd.DataFrame(holder, columns = cols, index = indices)

	file_id = str(time.time())[-7:]
	df.to_csv(f"./results/EstOpt-{batch_number}-{file_id}.csv")

	print (f'Batch number {batch_number} is done in {(time.time()-start)/60:.2f} minutes')
	start = time.time()
	batch_number += 1
	holder = []

	return holder, start, batch_number