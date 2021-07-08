import multiprocessing as mp
from multiprocessing import Process, Queue, Lock

def go_crazy(N = 10, q_out = None):
	"""A function for parallel processing"""
	gaopt_instance = GAOpt()
	for _ in range(N):
		q_out.put(gaopt_instance.optimize())

def GAOpt_parallel(N = 1000, batch_size = 100):

	results_queue = Queue()
	n_cores = mp.cpu_count()-2
	N_for_each_core = int(N/n_cores)
	mylock = Lock()

	# Creating and filling the pool
	pool = []
	for j in range (n_cores):
		worker = Process(target = go_crazy, args = (N_for_each_core, results_queue, ))
		pool.append(worker)

	print('starting processes...')
	for worker in pool:
		worker.start()

	all_samples = []
	done_workers = 0
	batch_number = 0

	cols = []
	# cols = list(GAOpt().create_bridge().keys())[1:]
	# Three elements were used in the original project
	for elem in range(3):
		# 20 years, biennially, binary representation --> 20 variables
		for step in range(20):
			cols.append(f'Eelem{elem}-{step}')

	start = time.time()
	while any(worker.is_alive() for worker in pool):

		while not results_queue.empty():
			sample = results_queue.get()

			if not sample is None:
				all_samples.append(sample)

		# Saving each batch
		if len(all_samples) > batch_size:
			batch_number += 1
			df = pd.DataFrame(all_samples[:batch_size], columns = cols)
			
			print (f'Batch number {batch_number} is done in {time.time()-start:.2f}')
			start = time.time()


			df.to_csv(f"./results/EstOpt-{batch_number}.csv")
			all_samples = all_samples[batch_size:]

	# Saving the last batch
	if len(all_samples) > 0:
		batch_number += 1
		df = pd.DataFrame(all_samples, columns = cols)
		df.to_csv(f"./results/EstOpt-{batch_number}.csv")
		print (f'Batch number {batch_number} is done')

	print('waiting for workers to join...')
	for worker in pool:
		worker.join()
	print('all workers are joined.\n')