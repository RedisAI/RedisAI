import redis

from includes import *
import os
from functools import wraps

'''
python -m RLTest --test tests_llapi.py --module path/to/redisai.so
'''


def ensure_test_module_loaded(f):
    @wraps(f)
    def wrapper(env, *args, **kwargs):
        goal_dir = os.path.join(os.getcwd(), "../module/LLAPI.so")
        TEST_MODULE_PATH = os.path.abspath(goal_dir)
        con = env.getConnection()
        modules = con.execute_command("MODULE", "LIST")
        if b'RAI_llapi' in [module[1] for module in modules]:
            return f(env, *args, **kwargs)
        try:
            ret = con.execute_command('MODULE', 'LOAD', TEST_MODULE_PATH)
            env.assertEqual(ret, b'OK')
            return f(env, *args, **kwargs)
        except Exception as e:
            env.assertFalse(True)
            env.debugPrint(str(e), force=True)
            return
    return wrapper


@ensure_test_module_loaded
def test_basic_check(env):

    con = env.getConnection()
    ret = con.execute_command("RAI_llapi.basic_check")
    env.assertEqual(ret, b'OK')


@ensure_test_module_loaded
def test_model_run_async(env):

    con = env.getConnection()
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    ret = con.execute_command("RAI_llapi.modelRun")
    env.assertEqual(ret, b'Async run success')


@ensure_test_module_loaded
def test_script_run_async(env):

    con = env.getConnection()
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    script_filename = os.path.join(test_data_path, 'script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    ret = con.execute_command('AI.SCRIPTSET', 'myscript{1}', DEVICE, 'TAG', 'version1', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command("RAI_llapi.scriptRun")
    env.assertEqual(ret, b'Async run success')
