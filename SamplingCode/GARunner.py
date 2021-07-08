import os
from subprocess import call, Popen, PIPE, STDOUT
import json
from _utils import load_indiana_assets


p = Popen("powershell python GAOpt.py", stdout=PIPE, stdin = PIPE)


assets = load_indiana_assets(4)


response = p.communicate(json.dumps(assets).encode('utf-8'))

print (response)

p.stdout.close()


print ('Done')



# for i in range (N):
# 	print (i, 'is called')
# 	Popen("powershell python GAOpt.py")
