import time
import os
import glob 
from subprocess import run

from __init__ import test_project_absroot

lines = []

def initenviroment():
	if not 0==run(['black','--version']):
		os.system("pip3 install black")
	if not 0==run(['isort','--version']):
		os.system("pip3 install isort")
def handlepyfile(py):
	global lines
	if not 0==run(['isort',py]):
		print("isort exception on ",str(py),time.ctime())
	if not 0==run(['black',py]):
		print("black exception on ",str(py),time.ctime())
	f = open(py,"r",encoding="utf-8")
	lines.append(len(f.readlines()))
	f.close()
	pass
def glob_dirs(base_dir):
	def access_pyfile(root):
		dirs = glob.glob("%s/*"%(root))
		for di in dirs:
			if os.path.isdir(di):
				access_pyfile(di)
			else:
				if di.endswith(".py") or di.endswith(".pyc") or di.endswith(".pyx"):
					handlepyfile(di)
	
	initenviroment()
	dirs = glob.glob("%s"%(base_dir))
	for di in dirs:
		access_pyfile(di)
	pass
if __name__ == '__main__':
	glob_dirs(test_project_absroot)
	print("total lines number is %d"%(len(lines)))
	f = open("./temp/debug.log","a",encoding="utf-8")
	f.write(str(lines))
	f.close()