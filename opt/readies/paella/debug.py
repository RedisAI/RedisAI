
import os

#----------------------------------------------------------------------------------------------

pydebug = os.environ.get('PYDEBUG', '')
if pydebug == '1' or pydebug == 'pudb':
    try:
        from pudb import set_trace as bb
    except ImportError:
        from pdb import set_trace as bb
elif pydebug == 'pdb':
    from pdb import set_trace as bb
else:
	def bb(): pass

#----------------------------------------------------------------------------------------------
