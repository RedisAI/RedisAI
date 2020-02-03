
import sys
from subprocess import Popen, PIPE

def eprint(*args, **kwargs):
	print(*args, file = sys.stderr, **kwargs)

def sh(cmd):
    return " ".join(Popen(cmd.split(), stdout=PIPE).communicate()[0].decode('utf-8').split("\n"))

