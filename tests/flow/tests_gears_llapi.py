import redis

from includes import *
from RLTest import Env
import os
from functools import wraps


def verify_gears_loaded(env):
    con = env.getConnection()
    modules = con.execute_command("MODULE", "LIST")
    if b'rg' in [module[1] for module in modules]:
        return True
    platform = paella.Platform()
    redisgears_dir = "{ROOT}/bin/{PLATFORM}/RedisGears".format(ROOT=ROOT, PLATFORM=platform.triplet())
    if not os.path.isdir(redisgears_dir):
        env.debugPrint("RedisGears directory does not exist", force=True)
        return False
    redisgears_path = os.path.join(redisgears_dir, 'redisgears.so')
    python_plugin_path = os.path.join(redisgears_dir, 'plugin/gears_python.so')
    try:
        ret = con.execute_command('MODULE', 'LOAD', redisgears_path, 'Plugin', python_plugin_path, 'CreateVenv', 0,
                                  'PythonInstallationDir', redisgears_dir)
        env.assertEqual(ret, b'OK')
    except Exception as e:
        env.debugPrint("RedisGears not loaded: "+str(e), force=True)
        return False
    return True


class TestModelExecuteFromGears:

    def __init__(self):
        self.env = Env()
        if not verify_gears_loaded(self.env):
            self.env.skip()
            return
        script = '''

import redisAI

def ModelRun_oldAPI(record):
    keys = ['a{1}', 'b{1}']
    tensors = redisAI.mgetTensorsFromKeyspace(keys)
    modelRunner = redisAI.createModelRunner('m{1}')
    redisAI.modelRunnerAddInput(modelRunner, 'a', tensors[0])
    redisAI.modelRunnerAddInput(modelRunner, 'b', tensors[1])
    redisAI.modelRunnerAddOutput(modelRunner, 'c')
    res = redisAI.modelRunnerRun(modelRunner)
    redisAI.setTensorInKey('c{1}', res[0])
    return "ModelRun_oldAPI_OK"

async def ModelRun_Async(record):
    keys = ['a{1}', 'b{1}']
    tensors = redisAI.mgetTensorsFromKeyspace(keys)
    modelRunner = redisAI.createModelRunner('m{1}')
    redisAI.modelRunnerAddInput(modelRunner, 'a', tensors[0])
    redisAI.modelRunnerAddInput(modelRunner, 'b', tensors[1])
    redisAI.modelRunnerAddOutput(modelRunner, 'c')
    res = await redisAI.modelRunnerRunAsync(modelRunner)
    redisAI.setTensorInKey('c_1{1}', res[0])
    return "ModelRun_Async_OK"

async def ModelRun_AsyncIgnoreInputNames(record):
    keys = ['a{1}', 'b{1}']
    tensors = redisAI.mgetTensorsFromKeyspace(keys)
    modelRunner = redisAI.createModelRunner('m{1}')
    redisAI.modelRunnerAddInput(modelRunner, 'input_name_in_model_definition_is_a', tensors[0])
    redisAI.modelRunnerAddInput(modelRunner, 'input_name_in_model_definition_is_b', tensors[1])
    redisAI.modelRunnerAddOutput(modelRunner, 'output_name_in_model_definition_is_c')
    res = await redisAI.modelRunnerRunAsync(modelRunner)
    redisAI.setTensorInKey('c_2{1}', res[0])
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
GB("CommandReader").map(ModelRun_AsyncIgnoreInputNames).register(trigger="ModelRun_Async_test3")
GB("CommandReader").map(ModelRun_AsyncRunError).register(trigger="ModelRun_AsyncRunError_test4")
    '''

        con = self.env.getConnection()
        ret = con.execute_command('rg.pyexecute', script)
        self.env.assertEqual(ret, b'OK')

        model_pb = load_file_content('graph.pb')
        ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TF', DEVICE,
                                  'INPUTS', 2, 'a', 'b', 'OUTPUTS', 1, 'mul', 'BLOB', model_pb)
        self.env.assertEqual(ret, b'OK')

        con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
        con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    def test_old_api(self):
        con = self.env.getConnection()
        ret = con.execute_command('rg.trigger', 'ModelRun_oldAPI_test1')
        self.env.assertEqual(ret[0], b'ModelRun_oldAPI_OK')
        values = con.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
        self.env.assertEqual(values, [b'4', b'9', b'4', b'9'])

    def test_async_run(self):
        con = self.env.getConnection()
        ret = con.execute_command('rg.trigger', 'ModelRun_Async_test2')
        self.env.assertEqual(ret[0], b'ModelRun_Async_OK')
        values = con.execute_command('AI.TENSORGET', 'c_1{1}', 'VALUES')
        self.env.assertEqual(values, [b'4', b'9', b'4', b'9'])

    def test_tf_ignore_inputs_names(self):
        con = self.env.getConnection()
        ret = con.execute_command('rg.trigger', 'ModelRun_Async_test3')
        self.env.assertEqual(ret[0], b'ModelRun_Async_OK')
        values = con.execute_command('AI.TENSORGET', 'c_2{1}', 'VALUES')
        self.env.assertEqual(values, [b'4', b'9', b'4', b'9'])

    def test_runtime_error(self):
        con = self.env.getConnection()
        ret = con.execute_command('rg.trigger', 'ModelRun_AsyncRunError_test4')
        # This should raise an exception
        self.env.assertEqual(str(ret[0]), "b'Must specify at least one target to fetch or execute.'")


class TestScriptExecuteFromGears:

    def __init__(self):
        self.env = Env()
        if not verify_gears_loaded(self.env):
            self.env.skip()
            return
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
    redisAI.setTensorInKey('c_1{1}', res[0])
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

        con = self.env.getConnection()
        ret = con.execute_command('rg.pyexecute', script)
        self.env.assertEqual(ret, b'OK')

        script = load_file_content("script.txt")
        ret = con.execute_command('AI.SCRIPTSTORE', 'myscript{1}', DEVICE, 'TAG', 'version1', 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
        self.env.assertEqual(ret, b'OK')
        ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
        self.env.assertEqual(ret, b'OK')
        ret = con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
        self.env.assertEqual(ret, b'OK')

    def test_old_api(self):
        con = self.env.getConnection()
        ret = con.execute_command('rg.trigger', 'ScriptRun_oldAPI_test1')
        self.env.assertEqual(ret[0], b'ScriptRun_oldAPI_OK')
        values = con.execute_command('AI.TENSORGET', 'c{1}', 'VALUES')
        self.env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    def test_async_execution(self):
        con = self.env.getConnection()
        ret = con.execute_command('rg.trigger', 'ScriptRun_Async_test2')
        self.env.assertEqual(ret[0], b'ScriptRun_Async_OK')
        values = con.execute_command('AI.TENSORGET', 'c_1{1}', 'VALUES')
        self.env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    def test_runtime_error(self):
        con = self.env.getConnection()
        ret = con.execute_command('rg.trigger', 'ScriptRun_AsyncRunError_test3')
        # This should raise an exception
        self.env.assertTrue(str(ret[0]).startswith("b'attempted to get undefined function"))


class TestDAGRunExecution:

    def __init__(self):
        self.env = Env()
        if not verify_gears_loaded(self.env):
            self.env.skip()
            return
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
    DAGRunner.OpsFromString('|> AI.MODELEXECUTE m{1} INPUTS 2 tensor_a tensor_b OUTPUTS 1 tensor_c |> AI.TENSORGET tensor_c')
    res = await DAGRunner.Run()
    redisAI.setTensorInKey('test5_res{1}', res[0])
    return "test5_OK"
        
GB("CommandReader").map(DAGRun_tensorSetTensorGet).register(trigger="DAGRun_test1")
GB("CommandReader").map(DAGRun_simpleModelRun).register(trigger="DAGRun_test2")
GB("CommandReader").map(DAGRun_simpleScriptRun).register(trigger="DAGRun_test3")
GB("CommandReader").map(DAGRun_scriptRunError).register(trigger="DAGRun_test4")
GB("CommandReader").map(DAGRun_addOpsFromString).register(trigger="DAGRun_test5")
    '''

        con = self.env.getConnection()
        ret = con.execute_command('rg.pyexecute', script)
        self.env.assertEqual(ret, b'OK')

        con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
        con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

        model_pb = load_file_content('graph.pb')
        ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TF', DEVICE,
                                  'INPUTS', 2, 'a', 'b', 'OUTPUTS', 1, 'mul', 'BLOB', model_pb)
        self.env.assertEqual(ret, b'OK')

        script = load_file_content('script.txt')
        ret = con.execute_command('AI.SCRIPTSTORE', 'myscript{1}', DEVICE, 'TAG', 'version1', 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
        self.env.assertEqual(ret, b'OK')

    def test_modelset_modelget_ops(self):
        con = self.env.getConnection()
        ret = con.execute_command('rg.trigger', 'DAGRun_test1')
        self.env.assertEqual(ret[0], b'test1_OK')
        values = con.execute_command('AI.TENSORGET', 'test1_res{1}', 'VALUES')
        self.env.assertEqual(values, [b'2', b'3', b'2', b'3'])

    def test_modelexecute_op(self):

        def multiple_executions(con):
            ret = con.execute_command('rg.trigger', 'DAGRun_test2')
            self.env.assertEqual(ret[0], b'test2_OK')
            values = con.execute_command('AI.TENSORGET', 'test2_res{1}', 'VALUES')
            self.env.assertEqual(values, [b'4', b'9', b'4', b'9'])

        run_test_multiproc(self.env, 500, multiple_executions)

    def test_scriptexecute_op(self):
        con = self.env.getConnection()
        ret = con.execute_command('rg.trigger', 'DAGRun_test3')
        self.env.assertEqual(ret[0], b'test3_OK')
        values = con.execute_command('AI.TENSORGET', 'test3_res{1}', 'VALUES')
        self.env.assertEqual(values, [b'4', b'6', b'4', b'6'])

    def test_scriptexecute_op_runtime_error(self):
        con = self.env.getConnection()
        ret = con.execute_command('rg.trigger', 'DAGRun_test4')
        # This should raise an exception
        self.env.assertTrue(str(ret[0]).startswith("b'attempted to get undefined function"))

    def test_build_dag_from_string(self):
        con = self.env.getConnection()
        ret = con.execute_command('rg.trigger', 'DAGRun_test5')
        self.env.assertEqual(ret[0], b'test5_OK')
        values = con.execute_command('AI.TENSORGET', 'test5_res{1}', 'VALUES')
        self.env.assertEqual(values, [b'4', b'9', b'4', b'9'])


class TestTensorCreate:

    def __init__(self):
        self.env = Env()
        if not verify_gears_loaded(self.env):
            self.env.skip()
            return
        script = '''

import redisAI
        
def TensorCreate_FromValues(record):
    tensor = redisAI.createTensorFromValues('DOUBLE', [2,2], [1.0, 2.0, 3.0, 4.0])
    redisAI.setTensorInKey('test1_res{1}', tensor)
    return "test1_OK"

def TensorCreate_FromBlob(record):
    tensor_blob = bytearray([5, 6, 7, 8])
    tensor = redisAI.createTensorFromBlob('INT8', [2,2], tensor_blob)
    redisAI.setTensorInKey('test2_res{1}', tensor)
    return "test2_OK"
        
GB("CommandReader").map(TensorCreate_FromValues).register(trigger="TensorCreate_FromValues_test1")
GB("CommandReader").map(TensorCreate_FromBlob).register(trigger="TensorCreate_FromBlob_test2")
    '''

        con = self.env.getConnection()
        ret = con.execute_command('rg.pyexecute', script)
        self.env.assertEqual(ret, b'OK')

    def test_create_tensor_from_values(self):
        con = self.env.getConnection()
        ret = con.execute_command('rg.trigger', 'TensorCreate_FromValues_test1')
        self.env.assertEqual(ret[0], b'test1_OK')
        values = con.execute_command('AI.TENSORGET', 'test1_res{1}', 'VALUES')
        self.env.assertEqual(values, [b'1', b'2', b'3', b'4'])

    def test_create_tensor_from_blob(self):
        con = self.env.getConnection()
        ret = con.execute_command('rg.trigger', 'TensorCreate_FromBlob_test2')
        self.env.assertEqual(ret[0], b'test2_OK')
        values = con.execute_command('AI.TENSORGET', 'test2_res{1}', 'VALUES')
        self.env.assertEqual(values, [5, 6, 7, 8])


def test_flatten_tensor_via_gears(env):
    if not verify_gears_loaded(env):
        env.skip()
        return
    script = '''

import redisAI
        
def FlattenTensor(record):
    
    tensor = redisAI.createTensorFromValues('DOUBLE', [2,2], [1.0, 2.0, 3.0, 4.0])
    tensor_as_list = redisAI.tensorToFlatList(tensor)
    if tensor_as_list != [1.0, 2.0, 3.0, 4.0]:
        return "ERROR failed to flatten tensor to list of doubles"
        
    tensor_blob = bytearray([5, 0, 6, 0, 7, 0, 8, 0])
    tensor = redisAI.createTensorFromBlob('UINT16', [2,2], tensor_blob)
    tensor_as_list = redisAI.tensorToFlatList(tensor)
    if tensor_as_list != [5, 6, 7, 8]:
        return "ERROR failed to flatten tensor to list of long long"
    return "test_OK"

        
GB("CommandReader").map(FlattenTensor).register(trigger="FlattenTensor_test")
    '''

    con = env.getConnection()
    ret = con.execute_command('rg.pyexecute', script)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('rg.trigger', 'FlattenTensor_test')
    env.assertEqual(ret[0], b'test_OK')


class TestExecuteOnnxModel:

    def __init__(self):
        self.env = Env()
        if not verify_gears_loaded(self.env):
            self.env.skip()
            return
        script = '''

import redisAI

def OnnxModelRunSync(record):
    input_tensor = redisAI.getTensorFromKey('mnist_input{1}')
    modelRunner = redisAI.createModelRunner('mnist{1}')
    redisAI.modelRunnerAddInput(modelRunner, 'input_name', input_tensor)
    redisAI.modelRunnerAddOutput(modelRunner, 'output_name')
    try:
        res = redisAI.modelRunnerRun(modelRunner)
    except Exception as e:
        raise e

async def OnnxModelRunAsync(record):
    input_tensor = redisAI.getTensorFromKey('mnist_input{1}')
    modelRunner = redisAI.createModelRunner('mnist{1}')
    redisAI.modelRunnerAddInput(modelRunner, 'input_name', input_tensor)
    redisAI.modelRunnerAddOutput(modelRunner, 'output_name')
    res = await redisAI.modelRunnerRunAsync(modelRunner)
    redisAI.setTensorInKey('mnist_output{1}', res[0])
    return "OnnxModelRun_OK"
        
GB("CommandReader").map(OnnxModelRunSync).register(trigger="OnnxModelRunSync_test1")
GB("CommandReader").map(OnnxModelRunAsync).register(trigger="OnnxModelRunAsync_test2")
    '''

        con = self.env.getConnection()
        ret = con.execute_command('rg.pyexecute', script)
        self.env.assertEqual(ret, b'OK')

        # Load onnx model and its input.
        model_pb = load_file_content('mnist.onnx')
        sample_raw = load_file_content('one.raw')
        ret = con.execute_command('AI.MODELSTORE', 'mnist{1}', 'ONNX', DEVICE, 'BLOB', model_pb)
        self.env.assertEqual(ret, b'OK')
        con.execute_command('AI.TENSORSET', 'mnist_input{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)

    def test_sync_run_error(self):
        con = self.env.getConnection()
        check_error_message(self.env, con, "Cannot execute onnxruntime model synchronously, use async execution instead",
                            'rg.trigger', 'OnnxModelRunSync_test1',
                            error_msg_is_substr=True)

    def test_async_run(self):
        con = self.env.getConnection()
        ret = con.execute_command('rg.trigger', 'OnnxModelRunAsync_test2')
        self.env.assertEqual(ret[0], b'OnnxModelRun_OK')
        values = con.execute_command('AI.TENSORGET', 'mnist_output{1}', 'VALUES')
        argmax = max(range(len(values)), key=lambda i: values[i])
        self.env.assertEqual(argmax, 1)
