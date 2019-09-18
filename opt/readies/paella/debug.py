
import os

#----------------------------------------------------------------------------------------------

if 'PYDEBUG' in os.environ:
    try:
        print('importing pudb')
        from pudb import set_trace as bb
        print('importing pudb: ok')
    except ImportError:
        from pdb import set_trace as bb
else:
	def bb(): pass

#----------------------------------------------------------------------------------------------
