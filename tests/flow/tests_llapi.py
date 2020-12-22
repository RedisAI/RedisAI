import redis

from includes import *
import os
from functools import wraps

'''
python -m RLTest --test tests_llapi.py --module path/to/redisai.so
'''

goal_dir = os.path.join(os.getcwd(), "../module/LLAPI.so")
TEST_MODULE_PATH = os.path.abspath(goal_dir)


def skip_if_gears_not_loaded(f):
    @wraps(f)
    def wrapper(env, *args, **kwargs):
        con = env.getConnection()
        modules = con.execute_command("MODULE", "LIST")
        if "rg" in [module[1] for module in modules]:
            return f(env, *args, **kwargs)
        try:
            redisgears_path = os.path.join(os.path.dirname(__file__), '../../../RedisGears/redisgears.so')
            ret = con.execute_command('MODULE', 'LOAD', redisgears_path)
            env.assertEqual(ret, b'OK')
            return f(env, *args, **kwargs)
        except Exception as e:
            env.debugPrint("skipping since RedisGears not loaded", force=True)
            return
    return wrapper


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
    ret = con.execute_command("RAI_llapi.modelRun")
    env.assertEqual(ret, b'Async run success')


@skip_if_gears_not_loaded
def test_model_run_async_via_gears(env):
    script = '''
import redisAI

async def RedisAIModelRun(record):
    keys = ['a{1}', 'b{1}']
    tensors = redisAI.mgetTensorsFromKeyspace(keys)
    modelRunner = redisAI.createModelRunner('m{1}')
    redisAI.modelRunnerAddInput(modelRunner, 'a', tensors[0])
    redisAI.modelRunnerAddInput(modelRunner, 'b', tensors[1])
    redisAI.modelRunnerAddOutput(modelRunner, 'mul')
    res = await redisAI.modelRunnerRunAsync(modelRunner)
    if len(res[1]) > 0:
        raise Exception(res[1][0])
    redisAI.setTensorInKey('c{1}', res[0][0])
    return "OK"

GB("CommandReader").map(RedisAIModelRun).register(trigger="ModelRunAsyncTest")
    '''
    con = env.getConnection()
    ret = con.execute_command('rg.pyexecute', script)
    env.assertEqual(ret, b'OK')

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 14)

    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT',
                        2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT',
                        2, 2, 'VALUES', 2, 3, 2, 3)

    ret = con.execute_command('rg.trigger', 'ModelRunAsyncTest')
    env.assertEqual(ret[0], b'OK')
    values = con.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'9', b'4', b'9'])
