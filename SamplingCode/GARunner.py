import os
import multiprocessing as mp

from subprocess import call, Popen

N = mp.cpu_count()
N = 1

for i in range (N):
	print (i, 'is called')
	Popen("powershell python GAOpt.py")
