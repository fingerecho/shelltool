
import os
import sys
from subprocess import run,PIPE
from getHash import hash
from test_hash_l import get_rand_str
files = os.listdir("./")

if __name__ == '__main__':
	input_s = get_rand_str()
	if len(sys.argv) > 1:
		input_s = sys.argv[1];print("input:",input_s)
	res = run(['node',files[0],'exec',input_s],shell=True,stdout=PIPE)
	out = res.stdout
	print("js-out:",out.decode('utf-8'))
	print("py-out:",hash(input_s))
