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
            'AI.SCRIPTSET', 'redis_scripts', DEVICE, 'SOURCE', script)
        self.env.assertEqual(ret, b'OK')
        # self.env.ensureSlaveSynced(self.con, self.env)

    # def test_float_get_set(self):
    #     self.con.execute_command('AI.SCRIPTRUN', 'redis_scripts', 'test_float_set_get', 'INPUTS', 'OUTPUTS', 'y')
    #     y = self.con.execute_command('AI.TENSORGET', 'y', 'meta' ,'VALUES')
    #     self.env.assertEqual(y, ["dtype", "FLOAT", "shape", [0, 1], "VALUES", "1.1"])

    def test_simple_test_set(self):
        self.con.execute_command(
            'AI.SCRIPTRUN', 'redis_scripts', 'test_set_key')
        self.env.assertEqual(b"1", self.con.get("x"))

    # def test_int_get_set(self):
    #     self.con.execute_command('AI.SCRIPTRUN', 'redis_scripts', 'test_int_set_get', 'OUTPUTS', 'y')
    #     y = self.con.execute_command('AI.TENSORGET', 'y', 'meta' ,'VALUES')
    #     self.env.assertEqual(y, ["dtype", "INT64", "shape", [0, 1], "VALUES", "1"] )