import os
import multiprocessing as mp
from subprocess import call, Popen, PIPE, STDOUT
import json
from _utils import load_indiana_assets


def run_indiana():

	n_cores = mp.cpu_count() - 10

	assets = load_indiana_assets(2)
	ids = list(assets.keys())
	sect_len = int(len(assets) / n_cores) + 1

	pool = []
	for i in range(n_cores):

		process_assets = {}

		final_index = min((i+1)*sect_len, len(assets))
		init_index = i*sect_len

		for k in ids[init_index : final_index]:
			process_assets[k] = assets[k]


		p = Popen("powershell python GAOpt.py", stdin = PIPE)
		p.stdin.write(json.dumps(process_assets).encode('utf-8'))

	print ('Done')





# for i in range (N):
# 	print (i, 'is called')
# 	Popen("powershell python GAOpt.py")

if __name__ == "__main__":
	run_indiana()
