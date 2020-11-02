import redis

from includes import *

'''
python -m RLTest --test tests_torchscript_extensions.py --module path/to/redisai.so
'''


class test_torch_script_extesions:

    def __init__(self):
        self.env = Env()
        if not TEST_PT:
            self.env.debugPrint("skipping {} since TEST_PT=0".format(sys._getframe().f_code.co_name), force=True)
            self.env.skip() 

        self.con = self.env.getConnect()
        script_filename = os.path.join(test_data_path, 'redis_scripts.txt')
        with open(script_filename, 'rb') as f:
            script = f.read()

        ret = self.con.execute_command('AI.SCRIPTSET', 'redis_scripts', DEVICE, 'SOURCE', script)
        self.env.assertEqual(ret, b'OK')
        self.env.ensureSlaveSynced(self.con, self.env)
    
    def test_int_get_set(self):
        self.con.execute_command('AI.SCRIPTRUN', 'redis_scripts', 'test_int_set_get', 'INPUTS', 'OUTPUTS', 'y')
        y = self.con.execute_command('AI.TENSORGET', 'y', 'meta' ,'VALUES')
        self.env.assertEqual(y, ["dtype", "INT64", "shape", [0, 1], "VALUES", "1"] )

    def test_float_get_set(self):
        self.con.execute_command('AI.SCRIPTRUN', 'redis_scripts', 'test_float_set_get', 'INPUTS', 'OUTPUTS', 'y')
        y = self.con.execute_command('AI.TENSORGET', 'y', 'meta' ,'VALUES')
        self.env.assertEqual(y, ["dtype", "FLOAT", "shape", [0, 1], "VALUES", "1.1"])
