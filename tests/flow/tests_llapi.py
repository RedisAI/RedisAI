import redis

from includes import *
import os
from functools import wraps

'''
python -m RLTest --test tests_llapi.py --module path/to/redisai.so
'''

def with_test_module(f):
    @wraps(f)
    def wrapper(env, *args, **kwargs):
        con = get_connection(env, '{1}')
        modules = con.execute_command("MODULE", "LIST")
        if b'RAI_llapi' in [module[1] for module in modules]:
            return f(env, *args, **kwargs)
        try:
            ret = con.execute_command('MODULE', 'LOAD', TESTMOD_PATH)
            env.assertEqual(ret, b'OK')
        except Exception as e:
            env.assertFalse(True)
            env.debugPrint(str(e), force=True)
            return
        return f(env, *args, **kwargs)
    return wrapper


@with_test_module
def test_basic_check(env):

    con = get_connection(env, '{1}')
    ret = con.execute_command("RAI_llapi.basic_check")
    env.assertEqual(ret, b'OK')


@with_test_module
def test_model_run_async(env):

    con = get_connection(env, '{1}')
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 2, 'a', 'b', 'OUTPUTS', 1, 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    ret = con.execute_command("RAI_llapi.modelRun")
    env.assertEqual(ret, b'Async run success')


@with_test_module
def test_script_run_async(env):

    con = get_connection(env, '{1}')
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    script_filename = os.path.join(test_data_path, 'script.txt')

    with open(script_filename, 'rb') as f:
        script = f.read()

    ret = con.execute_command('AI.SCRIPTSTORE', 'myscript{1}', DEVICE, 'TAG', 'version1', 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command("RAI_llapi.scriptRun")
    env.assertEqual(ret, b'Async run success')


@with_test_module
def test_dag_build_and_run(env):
    con = get_connection(env, '{1}')

    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT',
                        2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT',
                        2, 2, 'VALUES', 2, 3, 2, 3)
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'graph.pb')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()
    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 2, 'a', 'b', 'OUTPUTS', 1, 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    script_filename = os.path.join(test_data_path, 'script.txt')
    with open(script_filename, 'rb') as f:
        script = f.read()
    ret = con.execute_command('AI.SCRIPTSTORE', 'myscript{1}', DEVICE, 'TAG', 'version1', 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command("RAI_llapi.DAGrun")
    env.assertEqual(ret, b'DAG run success')

    # Run the DAG LLAPI test again with multi process test to ensure that there are no dead-locks
    def run_dag_llapi(con):
        con.execute_command("RAI_llapi.DAGrun")

    run_test_multiproc(env, '{1}', 500, run_dag_llapi)


@with_test_module
def test_dagrun_multidevice_resnet(env):
    con = get_connection(env, '{1}')

    model_name_0 = 'imagenet_model1:{1}'
    model_name_1 = 'imagenet_model2:{1}'
    script_name_0 = 'imagenet_script1:{1}'
    script_name_1 = 'imagenet_script2:{1}'
    inputvar = 'images'
    outputvar = 'output'
    image_key = 'image:{1}'
    temp_key1 = 'temp_key1:{1}'
    temp_key2_0 = 'temp_key2_0'
    temp_key2_1 = 'temp_key2_1'
    class_key_0 = 'output0:{1}'
    class_key_1 = 'output1:{1}'

    model_pb, script, labels, img = load_resnet_test_data()

    device_0 = 'CPU:1'
    device_1 = DEVICE

    ret = con.execute_command('AI.MODELSTORE', model_name_0, 'TF', device_0,
                              'INPUTS', 1, inputvar,
                              'OUTPUTS', 1, outputvar,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELSTORE', model_name_1, 'TF', device_1,
                              'INPUTS', 1, inputvar,
                              'OUTPUTS', 1, outputvar,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.SCRIPTSTORE', script_name_0, device_0, 'ENTRY_POINTS', 4, 'pre_process_3ch', 'pre_process_4ch', 'post_process', 'ensemble', 'SOURCE', script)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.SCRIPTSTORE', script_name_1, device_1, 'ENTRY_POINTS', 4, 'pre_process_3ch', 'pre_process_4ch', 'post_process', 'ensemble', 'SOURCE', script)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes())
    env.assertEqual(ret, b'OK')

    ret = con.execute_command("RAI_llapi.DAG_resnet")
    env.assertEqual(ret, b'DAG resnet success')
