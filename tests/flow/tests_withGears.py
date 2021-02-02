import redis

from includes import *
import os
from functools import wraps


def skip_if_gears_not_loaded(f):
    @wraps(f)
    def wrapper(env, *args, **kwargs):
        con = env.getConnection()
        modules = con.execute_command("MODULE", "LIST")
        if b'rg' in [module[1] for module in modules]:
            return f(env, *args, **kwargs)
        try:
            redisgears_path = os.path.join(os.path.dirname(__file__), 'bin/RedisGears/redisgears.so')
            python_plugin_path = os.path.join(os.path.dirname(__file__), 'bin/RedisGears/plugin/gears_python.so')
            python_env_path = os.path.join(os.path.dirname(__file__), 'bin/RedisGears')
            ret = con.execute_command('MODULE', 'LOAD', redisgears_path, 'Plugin', python_plugin_path, 'CreateVenv',
                                      0, 'PythonInstallationDir', python_env_path)
            env.assertEqual(ret, b'OK')
        except Exception as e:
            env.debugPrint(str(e), force=True)
            env.debugPrint("skipping since RedisGears not loaded", force=True)
            return
        return f(env, *args, **kwargs)
    return wrapper


@skip_if_gears_not_loaded
def test_ping_gears(env):

    script = '''
def ping(record):
    return "pong"
    
GB("CommandReader").map(ping).register(trigger="ping_test")
'''
    con = env.getConnection()
    ret = con.execute_command('rg.pyexecute', script)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('rg.trigger', 'ping_test')
    env.assertEqual(ret[0], b'pong')


@skip_if_gears_not_loaded
def ntest_model_run(env):
    script = '''

import redisAI

def ModelRun_oldAPI(record):
    keys = ['a{1}', 'b{1}']
    tensors = redisAI.mgetTensorsFromKeyspace(keys)
    modelRunner = redisAI.createModelRunner('m{1}')
    redisAI.modelRunnerAddInput(modelRunner, 'a', tensors[0])
    redisAI.modelRunnerAddInput(modelRunner, 'b', tensors[1])
    redisAI.modelRunnerAddOutput(modelRunner, 'mul')
    res = redisAI.modelRunnerRun(modelRunner)
    redisAI.setTensorInKey('c{1}', res[0])
    return "ModelRun_oldAPI_OK"

async def ModelRun_Async(record):
    keys = ['a{1}', 'b{1}']
    tensors = redisAI.mgetTensorsFromKeyspace(keys)
    modelRunner = redisAI.createModelRunner('m{1}')
    redisAI.modelRunnerAddInput(modelRunner, 'a', tensors[0])
    redisAI.modelRunnerAddInput(modelRunner, 'b', tensors[1])
    redisAI.modelRunnerAddOutput(modelRunner, 'mul')
    res = await redisAI.modelRunnerRunAsync(modelRunner)
    redisAI.setTensorInKey('c{1}', res[0])
    return "ModelRun_Async_OK"

async def ModelRun_AsyncRunError(record):
    try:
        keys = ['a{1}', 'b{1}']
        tensors = redisAI.mgetTensorsFromKeyspace(keys)
        modelRunner = redisAI.createModelRunner('m{1}')
        res = await redisAI.modelRunnerRunAsync(modelRunner)
        return "Error - Exception was not raised"
    except Exception as e:
        return e
        
GB("CommandReader").map(ModelRun_oldAPI).register(trigger="ModelRun_oldAPI_test1")
GB("CommandReader").map(ModelRun_Async).register(trigger="ModelRun_Async_test2")
GB("CommandReader").map(ModelRun_AsyncRunError).register(trigger="ModelRun_AsyncRunError_test3")
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

    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    ret = con.execute_command('rg.trigger', 'ModelRun_oldAPI_test1')
    env.assertEqual(ret[0], b'ModelRun_oldAPI_OK')
    values = con.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'9', b'4', b'9'])

    ret = con.execute_command('rg.trigger', 'ModelRun_Async_test2')
    env.assertEqual(ret[0], b'ModelRun_Async_OK')
    values = con.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'9', b'4', b'9'])

    ret = con.execute_command('rg.trigger', 'ModelRun_AsyncRunError_test3')
    # This should raise an exception
    env.assertEqual(ret[0].__str__(), "b'Must specify at least one target to fetch or execute.'")


@skip_if_gears_not_loaded
def ntest_script_run(env):
    script = '''

import redisAI

def ScriptRun_oldAPI(record):
    keys = ['a{1}', 'b{1}']
    tensors = redisAI.mgetTensorsFromKeyspace(keys)
    scriptRunner = redisAI.createScriptRunner('myscript{1}', 'bar')
    redisAI.scriptRunnerAddInput(scriptRunner, tensors[0])
    redisAI.scriptRunnerAddInput(scriptRunner, tensors[1])
    redisAI.scriptRunnerAddOutput(scriptRunner)
    res = redisAI.scriptRunnerRun(scriptRunner)
    redisAI.setTensorInKey('c{1}', res[0])
    return "ScriptRun_oldAPI_OK"

async def ScriptRun_Async(record):
    keys = ['a{1}', 'b{1}']
    tensors = redisAI.mgetTensorsFromKeyspace(keys)
    scriptRunner = redisAI.createScriptRunner('myscript{1}', 'bar')
    redisAI.scriptRunnerAddInput(scriptRunner, tensors[0])
    redisAI.scriptRunnerAddInput(scriptRunner, tensors[1])
    redisAI.scriptRunnerAddOutput(scriptRunner)
    res = await redisAI.scriptRunnerRunAsync(scriptRunner)
    redisAI.setTensorInKey('c{1}', res[0])
    return "ScriptRun_Async_OK"

async def ScriptRun_AsyncRunError(record):
    try:
        keys = ['a{1}', 'b{1}']
        tensors = redisAI.mgetTensorsFromKeyspace(keys)
        scriptRunner = redisAI.createScriptRunner('myscript{1}', 'bad_func')
        redisAI.scriptRunnerAddInput(scriptRunner, tensors[0])
        redisAI.scriptRunnerAddInput(scriptRunner, tensors[1])
        redisAI.scriptRunnerAddOutput(scriptRunner)
        res = await redisAI.scriptRunnerRunAsync(scriptRunner)
        return "Error - Exception was not raised"
    except Exception as e:
        return e
        
GB("CommandReader").map(ScriptRun_oldAPI).register(trigger="ScriptRun_oldAPI_test1")
GB("CommandReader").map(ScriptRun_Async).register(trigger="ScriptRun_Async_test2")
GB("CommandReader").map(ScriptRun_AsyncRunError).register(trigger="ScriptRun_AsyncRunError_test3")
    '''

    con = env.getConnection()
    ret = con.execute_command('rg.pyexecute', script)
    env.assertEqual(ret, b'OK')

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

    ret = con.execute_command('rg.trigger', 'ScriptRun_oldAPI_test1')
    env.assertEqual(ret[0], b'ScriptRun_oldAPI_OK')
    values = con.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    ret = con.execute_command('rg.trigger', 'ScriptRun_Async_test2')
    env.assertEqual(ret[0], b'ScriptRun_Async_OK')
    values = con.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    ret = con.execute_command('rg.trigger', 'ScriptRun_AsyncRunError_test3')
    # This should raise an exception
    error_string = b'attempted to get undefined function bad_func'
    env.assertEqual(str(ret[0])[:len(error_string)+2]+"'", "{}".format(error_string))


@skip_if_gears_not_loaded
def ntest_DAG_run_via_gears(env):
    script = '''

import redisAI

async def DAGRun_tensorSetTensorGet(record):
    keys = ['a{1}']
    tensors = redisAI.mgetTensorsFromKeyspace(keys)
    DAGRunner = redisAI.createDAGRunner()
    DAGRunner.TensorSet('tensor_a', tensors[0])
    DAGRunner.TensorGet('tensor_a')
    res = await DAGRunner.Run()
    redisAI.setTensorInKey('test1_res{1}', res[0])
    return "test1_OK"
    
async def DAGRun_simpleModelRun(record):
    
    keys = ['a{1}', 'b{1}']
    tensors = redisAI.mgetTensorsFromKeyspace(keys)
    DAGRunner = redisAI.createDAGRunner()
    DAGRunner.Input('tensor_a', tensors[0])
    DAGRunner.Input('tensor_b', tensors[1])
    DAGRunner.ModelRun(name='m{1}', inputs=['tensor_a', 'tensor_b'], outputs=['tensor_c'])
    DAGRunner.TensorGet('tensor_c')
    res = await DAGRunner.Run()
    redisAI.setTensorInKey('test2_res{1}', res[0])
    return "test2_OK"
    
async def DAGRun_simpleScriptRun(record):
    
    keys = ['a{1}', 'b{1}']
    tensors = redisAI.mgetTensorsFromKeyspace(keys)
    DAGRunner = redisAI.createDAGRunner()
    DAGRunner.Input('tensor_a', tensors[0])
    DAGRunner.Input('tensor_b', tensors[1])
    DAGRunner.ScriptRun(name='myscript{1}', func='bar', inputs=['tensor_a', 'tensor_b'], outputs=['tensor_c'])
    DAGRunner.TensorGet('tensor_c')
    res = await DAGRunner.Run()
    redisAI.setTensorInKey('test3_res{1}', res[0])
    return "test3_OK"
    
async def DAGRun_scriptRunError(record):
    
    keys = ['a{1}', 'b{1}']
    tensors = redisAI.mgetTensorsFromKeyspace(keys)
    DAGRunner = redisAI.createDAGRunner()
    DAGRunner.Input('tensor_a', tensors[0])
    DAGRunner.Input('tensor_b', tensors[1])
    DAGRunner.ScriptRun(name='myscript{1}', func='no_func', inputs=['tensor_a', 'tensor_b'], outputs=['tensor_c'])
    DAGRunner.TensorGet('tensor_c')
    try:
        res = await DAGRunner.Run()
    except Exception as e:
        return e
        
async def DAGRun_addOpsFromString(record):
    
    keys = ['a{1}', 'b{1}']
    tensors = redisAI.mgetTensorsFromKeyspace(keys)
    DAGRunner = redisAI.createDAGRunner()
    DAGRunner.Input('tensor_a', tensors[0]).Input('tensor_b', tensors[1])
    DAGRunner.OpsFromString('|> AI.MODELRUN m{1} INPUTS tensor_a tensor_b OUTPUTS tensor_c |> AI.TENSORGET tensor_c')
    res = await DAGRunner.Run()
    redisAI.setTensorInKey('test5_res{1}', res[0])
    return "test5_OK"
        
GB("CommandReader").map(DAGRun_tensorSetTensorGet).register(trigger="DAGRun_test1")
GB("CommandReader").map(DAGRun_simpleModelRun).register(trigger="DAGRun_test2")
GB("CommandReader").map(DAGRun_simpleScriptRun).register(trigger="DAGRun_test3")
GB("CommandReader").map(DAGRun_scriptRunError).register(trigger="DAGRun_test4")
GB("CommandReader").map(DAGRun_addOpsFromString).register(trigger="DAGRun_test5")
    '''

    con = env.getConnection()
    ret = con.execute_command('rg.pyexecute', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('rg.trigger', 'DAGRun_test1')
    env.assertEqual(ret[0], b'test1_OK')

    values = con.execute_command('AI.TENSORGET', 'test1_res{1}', 'VALUES')
    env.assertEqual(values, [b'2', b'3', b'2', b'3'])

    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT',
                        2, 2, 'VALUES', 2, 3, 2, 3)
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()
    ret = con.execute_command('AI.MODELSET', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 'a', 'b', 'OUTPUTS', 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('rg.trigger', 'DAGRun_test2')
    env.assertEqual(ret[0], b'test2_OK')

    values = con.execute_command('AI.TENSORGET', 'test2_res{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'9', b'4', b'9'])

    script_filename = os.path.join(test_data_path, 'script.txt')
    with open(script_filename, 'rb') as f:
        script = f.read()
    ret = con.execute_command('AI.SCRIPTSET', 'myscript{1}', DEVICE, 'TAG', 'version1', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('rg.trigger', 'DAGRun_test3')
    env.assertEqual(ret[0], b'test3_OK')

    values = con.execute_command('AI.TENSORGET', 'test3_res{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    ret = con.execute_command('rg.trigger', 'DAGRun_test4')
    # This should raise an exception

    error_string = b'attempted to get undefined function no_func'
    env.assertEqual(str(ret[0])[:len(error_string)+2]+"'", "{}".format(error_string))

    ret = con.execute_command('rg.trigger', 'DAGRun_test5')
    env.assertEqual(ret[0], b'test5_OK')

    values = con.execute_command('AI.TENSORGET', 'test5_res{1}', 'VALUES')
    env.assertEqual(values, [b'4', b'9', b'4', b'9'])
