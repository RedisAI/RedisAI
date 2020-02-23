
import sys
import inspect
import os.path

if (sys.version_info > (3, 0)):
	from .utils3 import *
else:
	from .utils2 import *

def current_filepath():
    return os.path.abspath(inspect.getfile(inspect.currentframe().f_back))
