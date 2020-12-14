import redis

from includes import *
import os

'''
python -m RLTest --test tests_llapi.py --module path/to/redisai.so
'''

#TEST_MODULE_PATH = "/home/alon/CLionProjects/RedisAI/bin/linux-x64-release/install-cpu/tests/redisai_testmodule/redisai_testmodule.so"
goal_dir = os.path.join(os.getcwd(), "../module/LLAPI.so")
TEST_MODULE_PATH = os.path.abspath(goal_dir)


def test_basic_check(env):

    con = env.getConnection()
    ret = con.execute_command("MODULE", "LOAD", TEST_MODULE_PATH)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command("RAI_llapi.basic_check")
    env.assertEqual(ret, b'OK')


def test_model_run_async(env):

    con = env.getConnection()
    ret = con.execute_command("MODULE", "LOAD", TEST_MODULE_PATH)
    env.assertEqual(ret, b'OK')

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    ret = con.execute_command("RAI_llapi.modelRunAsync")
    env.assertEqual(ret, b'Async Run Success')
