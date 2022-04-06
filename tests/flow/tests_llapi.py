import redis

from includes import *
import os
from functools import wraps
from RLTest import Env

'''
python -m RLTest --test tests_llapi.py --module path/to/redisai.so
'''

def with_test_module(f):
    @wraps(f)
    def wrapper(env, *args, **kwargs):
        env = Env(moduleArgs='THREADS_PER_QUEUE 8')
        con = get_connection(env, '{1}')
        modules = con.execute_command("MODULE", "LIST")
        if b'RAI_llapi' in [module[1] for module in modules]:
            return f(env, *args, **kwargs)
        try:
            TESTMOD_PATH = "/home/alon/Code/RedisAI/bin/linux-x64-release/src/tests/module/testmod.so"
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

    # This command in test module runs the model twice - one run returns with an error and the second with success.
    ret = con.execute_command("RAI_llapi.modelRun")
    env.assertEqual(ret, b'Async run success')

    # Check that statistics were saved properly
    info = info_to_dict(con.execute_command('AI.INFO', 'm{1}'))
    env.assertEqual(info['key'], 'm{1}')
    env.assertEqual(info['type'], 'MODEL')
    env.assertEqual(info['backend'], 'TF')
    env.assertEqual(info['device'], DEVICE)
    env.assertGreater(info['duration'], 0)
    env.assertEqual(info['samples'], 2)
    env.assertEqual(info['calls'], 2)
    env.assertEqual(info['errors'], 1)


@with_test_module
def test_script_run_async(env):

    con = get_connection(env, '{1}')
    script = load_file_content('script.txt')

    ret = con.execute_command('AI.SCRIPTSTORE', 'myscript{1}', DEVICE, 'TAG', 'version1', 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    env.assertEqual(ret, b'OK')

    # This command in test module runs the script twice - onc run returns with an error and the second with success.
    ret = con.execute_command("RAI_llapi.scriptRun")
    env.assertEqual(ret, b'Async run success')

    # Check that statistics were saved properly
    info = info_to_dict(con.execute_command('AI.INFO', 'myscript{1}'))
    env.assertEqual(info['key'], 'myscript{1}')
    env.assertEqual(info['type'], 'SCRIPT')
    env.assertEqual(info['backend'], 'TORCH')
    env.assertEqual(info['device'], DEVICE)
    env.assertGreater(info['duration'], 0)
    env.assertEqual(info['calls'], 2)
    env.assertEqual(info['errors'], 1)


@with_test_module
def test_dag_build_and_run(env):
    con = get_connection(env, '{1}')

    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)
    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    model_pb = load_file_content('graph.pb')
    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 2, 'a', 'b', 'OUTPUTS', 1, 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    script = load_file_content('script.txt')
    ret = con.execute_command('AI.SCRIPTSTORE', 'myscript{1}', DEVICE, 'TAG', 'version1', 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command("RAI_llapi.DAGrun")
    env.assertEqual(ret, b'DAG run success')

    # Run the DAG LLAPI test again with multiprocess test to ensure that there are no deadlocks
    executions_num = 100 if not VALGRIND else 10

    def run_dag_llapi(con, i):
        con.execute_command("RAI_llapi.DAGrun")

    run_test_multiproc(env, '{1}', executions_num, run_dag_llapi)

    # Check that statistics were saved properly - in every execution the model run (successfully) twice.
    # Every run is over two samples (the dim[0] of the input tensors).
    info = info_to_dict(con.execute_command('AI.INFO', 'm{1}'))
    env.assertEqual(info['key'], 'm{1}')
    env.assertEqual(info['type'], 'MODEL')
    env.assertEqual(info['backend'], 'TF')
    env.assertEqual(info['device'], DEVICE)
    env.assertGreater(info['duration'], executions_num)
    env.assertEqual(info['samples'], 4*(executions_num + 1))
    env.assertEqual(info['calls'], 2*(executions_num + 1))
    env.assertEqual(info['errors'], 0)

    # Check that statistics were saved properly - in every execution the script run twice - once successfully
    # and once with an error.
    info = info_to_dict(con.execute_command('AI.INFO', 'myscript{1}'))
    env.assertEqual(info['key'], 'myscript{1}')
    env.assertEqual(info['type'], 'SCRIPT')
    env.assertEqual(info['backend'], 'TORCH')
    env.assertEqual(info['device'], DEVICE)
    env.assertGreater(info['duration'], executions_num)
    env.assertEqual(info['calls'], 2*(executions_num + 1))
    env.assertEqual(info['errors'], executions_num + 1)


@with_test_module
def test_dagrun_multidevice_resnet(env):
    con = get_connection(env, '{1}')

    model_name_0 = 'imagenet_model1:{1}'
    model_name_1 = 'imagenet_model2:{1}'
    script_name_0 = 'imagenet_script1:{1}'
    input_var = 'images'
    output_var = 'output'
    image_key = 'image:{1}'

    model_pb, script, labels, img = load_resnet_test_data()

    device_0 = 'CPU:1'
    device_1 = DEVICE

    ret = con.execute_command('AI.MODELSTORE', model_name_0, 'TF', device_0,
                              'INPUTS', 1, input_var,
                              'OUTPUTS', 1, output_var,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELSTORE', model_name_1, 'TF', device_1,
                              'INPUTS', 1, input_var,
                              'OUTPUTS', 1, output_var,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.SCRIPTSTORE', script_name_0, DEVICE, 'ENTRY_POINTS', 4, 'pre_process_3ch',
                              'pre_process_4ch', 'post_process', 'ensemble', 'SOURCE', script)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes())
    env.assertEqual(ret, b'OK')

    ret = con.execute_command("RAI_llapi.DAG_resnet")
    env.assertEqual(ret, b'DAG resnet success')

    # Check that statistics were saved properly - in every execution both model run once, each on a different
    # device, and the script run 3 times.
    info = info_to_dict(con.execute_command('AI.INFO', model_name_0))
    env.assertEqual(info['key'], model_name_0)
    env.assertEqual(info['type'], 'MODEL')
    env.assertEqual(info['backend'], 'TF')
    env.assertEqual(info['device'], device_0)
    env.assertGreater(info['duration'], 0)
    env.assertEqual(info['samples'], 1)
    env.assertEqual(info['calls'], 1)
    env.assertEqual(info['errors'], 0)

    info = info_to_dict(con.execute_command('AI.INFO', model_name_1))
    env.assertEqual(info['key'], model_name_1)
    env.assertEqual(info['type'], 'MODEL')
    env.assertEqual(info['backend'], 'TF')
    env.assertEqual(info['device'], device_1)
    env.assertGreater(info['duration'], 0)
    env.assertEqual(info['samples'], 1)
    env.assertEqual(info['calls'], 1)
    env.assertEqual(info['errors'], 0)

    info = info_to_dict(con.execute_command('AI.INFO', script_name_0))
    env.assertEqual(info['key'], script_name_0)
    env.assertEqual(info['type'], 'SCRIPT')
    env.assertEqual(info['backend'], 'TORCH')
    env.assertEqual(info['device'], DEVICE)
    env.assertGreater(info['duration'], 0)
    env.assertEqual(info['calls'], 3)
    env.assertEqual(info['errors'], 0)


@with_test_module
def test_tensor_create(env):
    con = get_connection(env, '{1}')
    ret = con.execute_command("RAI_llapi.CreateTensor")
    env.assertEqual(ret, b'create tensor test success')
    ret = con.execute_command("RAI_llapi.ConcatenateTensors")
    env.assertEqual(ret, b'concatenate tensors test success')
    ret = con.execute_command("RAI_llapi.SliceTensor")
    env.assertEqual(ret, b'slice tensor test success')
