import math
import redis

from includes import *
from RLTest import Env

'''
python -m RLTest --test tests_torchscript_extensions.py --module path/to/redisai.so
'''


class test_torch_script_extesions:

    def __init__(self):
        self.env = Env()
        if not TEST_PT:
            self.env.debugPrint("skipping {} since TEST_PT=0".format(
                sys._getframe().f_code.co_name), force=True)
            self.env.skip()

        self.con = self.env.getConnection()
        test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
        script_filename = os.path.join(test_data_path, 'redis_scripts.py')
        with open(script_filename, 'rb') as f:
            script = f.read()

        ret = self.con.execute_command(
            'AI.SCRIPTSET', 'redis_scripts{1}', DEVICE, 'SOURCE', script)
        self.env.assertEqual(ret, b'OK')
        # self.env.ensureSlaveSynced(self.con, self.env)

    def test_redis_error(self):
        try:
            self.con.execute_command(
            'AI.SCRIPTRUN', 'redis_scripts', 'test_redis_error')
            self.env.assertTrue(False)
        except:
            pass
        
    def test_simple_test_set(self):
        self.con.execute_command(
            'AI.SCRIPTRUN', 'redis_scripts{1}', 'test_set_key')
        self.env.assertEqual(b"1", self.con.get("x{1}"))

    def test_int_set_get(self):
        self.con.execute_command('AI.SCRIPTRUN', 'redis_scripts{1}', 'test_int_set_get', 'OUTPUTS', 'y{1}')
        y = self.con.execute_command('AI.TENSORGET', 'y{1}', 'meta' ,'VALUES')
        self.env.assertEqual(y, [b"dtype", b"INT64", b"shape", [], b"values", [1]] )

    def test_int_set_incr(self):
        self.con.execute_command('AI.SCRIPTRUN', 'redis_scripts{1}', 'test_int_set_incr', 'OUTPUTS', 'y{1}')
        y = self.con.execute_command('AI.TENSORGET', 'y{1}', 'meta' ,'VALUES')
        self.env.assertEqual(y, [b"dtype", b"INT64", b"shape", [], b"values", [2]] )

    def test_float_get_set(self):
        self.con.execute_command('AI.SCRIPTRUN', 'redis_scripts{1}', 'test_float_set_get', 'OUTPUTS', 'y{1}')
        y = self.con.execute_command('AI.TENSORGET', 'y{1}', 'meta' ,'VALUES')
        self.env.assertEqual(y[0], b"dtype")
        self.env.assertEqual(y[1], b"FLOAT")
        self.env.assertEqual(y[2], b"shape")
        self.env.assertEqual(y[3], [])
        self.env.assertEqual(y[4], b"values")
        self.env.assertAlmostEqual(float(y[5][0]), 1.1, 0.1)
    
    def test_int_list(self):
        self.con.execute_command('AI.SCRIPTRUN', 'redis_scripts{1}', 'test_int_list', 'OUTPUTS', 'y{1}')
        y = self.con.execute_command('AI.TENSORGET', 'y{1}', 'meta' ,'VALUES')
        self.env.assertEqual(y, [b"dtype", b"INT64", b"shape", [2, 1], b"values", [1, 2]] )

    def test_hash(self):
        self.con.execute_command('AI.SCRIPTRUN', 'redis_scripts{1}', 'test_hash', 'OUTPUTS', 'y{1}')
        y = self.con.execute_command('AI.TENSORGET', 'y{1}', 'meta' ,'VALUES')
        self.env.assertEqual(y, [b"dtype", b"INT64", b"shape", [2, 1], b"values", [1, 2]] )