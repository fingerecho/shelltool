
import os
import sys
import string
from subprocess import run,PIPE
from getHash import hash
from random import randrange
from datetime import datetime

files = os.listdir("./")

def get_rand_str():
	lens = randrange(0,30)
	results = ""
	for i in range(lens):
		results+= string.printable[randrange(0,99)]
	#return results
	#return str(datetime.now())
	return datetime.now().strftime("%Y-%d-%M")

def run_test_str(input_s):
	res = run(['node',files[0],'exec',input_s],shell=True,stdout=PIPE)
	out = res.stdout.decode("utf-8").strip("\n");print(out)
	p_out = hash(input_s);print(p_out)
	if out == p_out:
		print("equal")
	else:
		print("not equal")
	assert out == p_out

def run_test():
	for i in range(0,100):
		run_test_str(get_rand_str())
	pass


if __name__ == '__main__':
	run_test()