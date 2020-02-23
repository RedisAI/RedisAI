
import sys
from subprocess import Popen, PIPE

def eprint(*args, **kwargs):
	print >> sys.stderr, ' '.join(map(lambda x: "%s" % x, args))

def sh(cmd):
    return " ".join(Popen(cmd.split(), stdout=PIPE).communicate()[0].split("\n"))
