import shutil
import sys
import os
import subprocess
import redis
from includes import *
from RLTest import Env

'''
python -m RLTest --test tests_onnx.py --module path/to/redisai.so
'''


def test_onnx_modelrun_mnist(env):
    if not TEST_ONNX:
        env.debugPrint("skipping {} since TEST_ONNX=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')
    model_pb = load_file_content('mnist.onnx')
    wrong_model_pb = load_file_content('graph.pb')
    sample_raw = load_file_content('one.raw')

    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'ONNX', DEVICE, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 16)
    env.assertEqual(ret[5], b'')
    env.assertEqual(len(ret[11]), 1)
    env.assertEqual(len(ret[13]), 1)

    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'ONNX', DEVICE, 'TAG', 'version:2', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 16)
    # TODO: enable me. CI is having issues on GPU asserts of ONNX and CPU
    if DEVICE == "CPU":
        env.assertEqual(ret[1], b'ONNX')
        env.assertEqual(ret[3], b'CPU')
    env.assertEqual(ret[5], b'version:2')
    env.assertEqual(len(ret[11]), 1)
    env.assertEqual(len(ret[13]), 1)

    check_error_message(env, con, "No graph was found in the protobuf.",
                        'AI.MODELSTORE', 'm{1}', 'ONNX', DEVICE, 'BLOB', wrong_model_pb)

    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)

    check_error_message(env, con, "Number of keys given as INPUTS here does not match model definition",
                        'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 3, 'a{1}', 'b{1}', 'c{1}', 'OUTPUTS', 'c{1}')

    con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 1, 'a{1}', 'OUTPUTS', 1, 'b{1}')

    ensureSlaveSynced(con, env)

    values = con.execute_command('AI.TENSORGET', 'b{1}', 'VALUES')
    argmax = max(range(len(values)), key=lambda i: values[i])
    env.assertEqual(argmax, 1)

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        values2 = con2.execute_command('AI.TENSORGET', 'b{1}', 'VALUES')
        env.assertEqual(values2, values)


def test_onnx_string_tensors(env):
    if not TEST_ONNX:
        env.debugPrint("skipping {} since TEST_ONNX=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')
    model_pb = load_file_content('identity_string.onnx')
    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'ONNX', DEVICE, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    # Execute onnx model whose input is string tensor with shape [2,2], that outputs the input
    string_tensor_blob = b'input11\0input12\0input21\0input22\0'
    con.execute_command('AI.TENSORSET', 'in_tensor{1}', 'STRING', 2, 2, 'BLOB', string_tensor_blob)
    ret = con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 1, 'in_tensor{1}', 'OUTPUTS', 1, 'out_tensor{1}')
    env.assertEqual(ret, b'OK')

    _, tensor_dtype, _, tensor_dim, _, tensor_values = con.execute_command('AI.TENSORGET', 'out_tensor{1}', 'META', 'VALUES')
    env.assertEqual(tensor_dtype, b'STRING')
    env.assertEqual(tensor_dim, [2, 2])
    env.assertEqual(tensor_values, [b'input11', b'input12', b'input21', b'input22'])

    if env.useSlaves:
        ensureSlaveSynced(con, env)
        slave_con = env.getSlaveConnection()
        slave_tensor_values = slave_con.execute_command('AI.TENSORGET', 'out_tensor{1}', 'VALUES')
        env.assertEqual(tensor_values, slave_tensor_values)


def test_onnx_string_tensors_batching(env):
    if not TEST_ONNX:
        env.debugPrint("skipping {} since TEST_ONNX=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')
    model_pb = load_file_content('identity_string.onnx')
    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'ONNX', DEVICE, 'BATCHSIZE', 2, 'MINBATCHSIZE', 2,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    con.execute_command('AI.TENSORSET', 'first_batch{1}', 'STRING', 1, 2, 'VALUES', 'this is\0', 'the first batch\0')
    con.execute_command('AI.TENSORSET', 'second_batch{1}', 'STRING', 1, 2, 'VALUES', 'that is\0', 'the second batch\0')

    def run():
        con2 = get_connection(env, '{1}')
        con2.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 1, 'first_batch{1}', 'OUTPUTS', 1, 'first_output{1}')

    t = threading.Thread(target=run)
    t.start()

    con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 1, 'second_batch{1}', 'OUTPUTS', 1, 'second_output{1}')
    t.join()

    out_values = con.execute_command('AI.TENSORGET', 'first_batch{1}', 'VALUES')
    env.assertEqual(out_values, [b'this is', b'the first batch'])
    out_values = con.execute_command('AI.TENSORGET', 'second_batch{1}', 'VALUES')
    env.assertEqual(out_values, [b'that is', b'the second batch'])


def test_onnx_modelrun_batchdim_mismatch(env):
    if not TEST_ONNX:
        env.debugPrint("skipping {} since TEST_ONNX=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')
    model_pb = load_file_content('batchdim_mismatch.onnx')

    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'ONNX', DEVICE, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 3, 'VALUES', 1, 1, 1)
    con.execute_command('AI.TENSORSET', 'b{1}', 'FLOAT', 2, 'VALUES', 1, 1)

    check_error_message(env, con, "Got invalid dimensions for input: 0 for the following indices  index: 0 Got: 3"
                                  " Expected: 2  Please fix either the inputs or the model.",
                        'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 2, 'a{1}', 'b{1}', 'OUTPUTS', 2, 'c{1}', 'd{1}')


def test_onnx_modelrun_mnist_autobatch(env):
    if not TEST_ONNX:
        return

    con = get_connection(env, '{1}')
    model_pb = load_file_content('mnist_batched.onnx')
    sample_raw = load_file_content('one.raw')

    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'ONNX', 'CPU',
                              'BATCHSIZE', 2, 'MINBATCHSIZE', 2, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELGET', 'm{1}', 'META')
    env.assertEqual(len(ret), 16)
    # TODO: enable me. CI is having issues on GPU asserts of ONNX and CPU
    if DEVICE == "CPU":
        env.assertEqual(ret[1], b'ONNX')
        env.assertEqual(ret[3], b'CPU')
    env.assertEqual(ret[5], b'')
    env.assertEqual(ret[7], 2)
    env.assertEqual(ret[9], 2)
    env.assertEqual(len(ret[11]), 1)
    env.assertEqual(len(ret[13]), 1)
    env.assertEqual(ret[15], 0)

    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)
    con.execute_command('AI.TENSORSET', 'c{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)

    ensureSlaveSynced(con, env)

    def run():
        con = get_connection(env, '{1}')
        con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 1, 'c{1}', 'OUTPUTS', 1, 'd{1}')

    t = threading.Thread(target=run)
    t.start()

    con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 1, 'a{1}', 'OUTPUTS', 1, 'b{1}')
    t.join()

    ensureSlaveSynced(con, env)

    import time
    time.sleep(1)

    values = con.execute_command('AI.TENSORGET', 'b{1}', 'VALUES')
    argmax = max(range(len(values)), key=lambda i: values[i])

    env.assertEqual(argmax, 1)

    values = con.execute_command('AI.TENSORGET', 'd{1}', 'VALUES')
    argmax = max(range(len(values)), key=lambda i: values[i])

    env.assertEqual(argmax, 1)


def test_onnx_modelrun_iris(env):
    if not TEST_ONNX:
        env.debugPrint("skipping {} since TEST_ONNX=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')

    linear_model = load_file_content('linear_iris.onnx')
    logreg_model = load_file_content('logreg_iris.onnx')

    ret = con.execute_command('AI.MODELSTORE', 'linear{1}', 'ONNX', DEVICE, 'BLOB', linear_model)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.MODELSTORE', 'logreg{1}', 'ONNX', DEVICE, 'BLOB', logreg_model)
    env.assertEqual(ret, b'OK')

    con.execute_command('AI.TENSORSET', 'features{1}', 'FLOAT', 1, 4, 'VALUES', 5.1, 3.5, 1.4, 0.2)

    ensureSlaveSynced(con, env)

    con.execute_command('AI.MODELEXECUTE', 'linear{1}', 'INPUTS', 1, 'features{1}', 'OUTPUTS', 1, 'linear_out{1}')
    con.execute_command('AI.MODELEXECUTE', 'logreg{1}', 'INPUTS', 1, 'features{1}', 'OUTPUTS', 2, 'logreg_out{1}', 'logreg_probs{1}')

    ensureSlaveSynced(con, env)

    linear_out = con.execute_command('AI.TENSORGET', 'linear_out{1}', 'VALUES')
    logreg_out = con.execute_command('AI.TENSORGET', 'logreg_out{1}', 'VALUES')

    env.assertEqual(float(linear_out[0]), -0.090524077415466309)
    env.assertEqual(logreg_out[0], 0)

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        linear_out2 = con2.execute_command('AI.TENSORGET', 'linear_out{1}', 'VALUES')
        logreg_out2 = con2.execute_command('AI.TENSORGET', 'logreg_out{1}', 'VALUES')
        env.assertEqual(linear_out, linear_out2)
        env.assertEqual(logreg_out, logreg_out2)


def test_onnx_modelinfo(env):
    if not TEST_ONNX:
        env.debugPrint("skipping {} since TEST_ONNX=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')
    linear_model = load_file_content('linear_iris.onnx')

    ret = con.execute_command('AI.MODELSTORE', 'linear{1}', 'ONNX', DEVICE, 'BLOB', linear_model)
    env.assertEqual(ret, b'OK')

    model_serialized_master = con.execute_command('AI.MODELGET', 'linear{1}', 'META')
    con.execute_command('AI.TENSORSET', 'features{1}', 'FLOAT', 1, 4, 'VALUES', 5.1, 3.5, 1.4, 0.2)

    ensureSlaveSynced(con, env)

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        model_serialized_slave = con2.execute_command('AI.MODELGET', 'linear{1}', 'META')
        env.assertEqual(len(model_serialized_master), len(model_serialized_slave))
    previous_duration = 0
    for call in range(1, 10):
        res = con.execute_command('AI.MODELEXECUTE', 'linear{1}', 'INPUTS', 1, 'features{1}', 'OUTPUTS', 1, 'linear_out{1}')
        env.assertEqual(res, b'OK')
        ensureSlaveSynced(con, env)

        info = con.execute_command('AI.INFO', 'linear{1}')
        info_dict_0 = info_to_dict(info)

        env.assertEqual(info_dict_0['key'], 'linear{1}')
        env.assertEqual(info_dict_0['type'], 'MODEL')
        env.assertEqual(info_dict_0['backend'], 'ONNX')
        env.assertEqual(info_dict_0['device'], DEVICE)
        env.assertTrue(info_dict_0['duration'] > previous_duration)
        env.assertEqual(info_dict_0['samples'], call)
        env.assertEqual(info_dict_0['calls'], call)
        env.assertEqual(info_dict_0['errors'], 0)

        previous_duration = info_dict_0['duration']

    res = con.execute_command('AI.INFO', 'linear{1}', 'RESETSTAT')
    env.assertEqual(res, b'OK')

    info = con.execute_command('AI.INFO', 'linear{1}')
    info_dict_0 = info_to_dict(info)
    env.assertEqual(info_dict_0['duration'], 0)
    env.assertEqual(info_dict_0['samples'], 0)
    env.assertEqual(info_dict_0['calls'], 0)
    env.assertEqual(info_dict_0['errors'], 0)


def test_onnx_modelrun_disconnect(env):
    if not TEST_ONNX:
        env.debugPrint("skipping {} since TEST_ONNX=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')
    linear_model = load_file_content('linear_iris.onnx')

    ret = con.execute_command('AI.MODELSTORE', 'linear{1}', 'ONNX', DEVICE, 'BLOB', linear_model)
    env.assertEqual(ret, b'OK')

    model_serialized_master = con.execute_command('AI.MODELGET', 'linear{1}', 'META')
    con.execute_command('AI.TENSORSET', 'features{1}', 'FLOAT', 1, 4, 'VALUES', 5.1, 3.5, 1.4, 0.2)

    ensureSlaveSynced(con, env)

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        model_serialized_slave = con2.execute_command('AI.MODELGET', 'linear{1}', 'META')
        env.assertEqual(len(model_serialized_master), len(model_serialized_slave))

    ret = send_and_disconnect(('AI.MODELEXECUTE', 'linear{1}', 'INPUTS', 1, 'features{1}', 'OUTPUTS', 1, 'linear_out{1}'), con)
    env.assertEqual(ret, None)


def tests_onnx_info(env):
    if not TEST_ONNX:
        env.debugPrint("skipping {} since TEST_ONNX=0".format(sys._getframe().f_code.co_name), force=True)
        return
    con = get_connection(env, '{1}')

    backends_info = get_info_section(con, 'backends_info')
    env.assertFalse('ai_onnxruntime_version' in backends_info)

    linear_model = load_file_content('linear_iris.onnx')
    con.execute_command('AI.MODELSTORE', 'linear{1}', 'ONNX', DEVICE, 'BLOB', linear_model)

    backends_info = get_info_section(con, 'backends_info')
    env.assertTrue('ai_onnxruntime_version' in backends_info)


def test_parallelism():
    env = Env(moduleArgs='INTRA_OP_PARALLELISM 1 INTER_OP_PARALLELISM 1')
    if not TEST_ONNX:
        env.debugPrint("skipping {} since TEST_ONNX=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = get_connection(env, '{1}')
    model_pb = load_file_content('mnist.onnx')
    sample_raw = load_file_content('one.raw')

    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'ONNX', DEVICE, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)

    con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 1, 'a{1}', 'OUTPUTS', 1, 'b{1}')
    ensureSlaveSynced(con, env)
    values = con.execute_command('AI.TENSORGET', 'b{1}', 'VALUES')
    argmax = max(range(len(values)), key=lambda i: values[i])
    env.assertEqual(argmax, 1)

    load_time_config = get_info_section(con, 'load_time_configs')
    env.assertEqual(load_time_config["ai_inter_op_parallelism"], "1")
    env.assertEqual(load_time_config["ai_intra_op_parallelism"], "1")

    env = Env(moduleArgs='INTRA_OP_PARALLELISM 2 INTER_OP_PARALLELISM 2')
    load_time_config = get_info_section(con, 'load_time_configs')
    env.assertEqual(load_time_config["ai_inter_op_parallelism"], "2")
    env.assertEqual(load_time_config["ai_intra_op_parallelism"], "2")


class TestOnnxKillSwitch:

    def __init__(self):
        self.threads_per_queue = 8
        self.env = Env(moduleArgs='THREADS_PER_QUEUE '+str(self.threads_per_queue)+' MODEL_EXECUTION_TIMEOUT 1000')
        con = get_connection(self.env, '{1}')
        model_with_inf_loop = load_file_content("model_with_infinite_loop.onnx")
        ret = con.execute_command('AI.MODELSTORE', 'inf_loop_model{1}', 'ONNX', DEVICE, 'BLOB', model_with_inf_loop)
        self.env.assertEqual(ret, b'OK')

        # Set tensors according to the model inputs. This model consists of two operations to type 'Identity'
        # (i.e., just output the input), where the second op is wrapped with another op of type 'Loop'. Overall, this
        # model runs a very large number of iterations without doing anything, until it is caught with the kill switch.
        con.execute_command('AI.TENSORSET', 'iterations{1}', 'INT64', 1, 'VALUES', 9223372036854775807)
        con.execute_command('AI.TENSORSET', 'loop_cond{1}', 'BOOL', 1, 'VALUES', 1)
        con.execute_command('AI.TENSORSET', 'loop_input{1}', 'FLOAT', 1, 'VALUES', 42)
        con.execute_command('AI.TENSORSET', 'outer_scope_input{1}', 'FLOAT', 1, 'VALUES', 42)

    def test_basic(self):
        con = get_connection(self.env, '{1}')
        check_error_message(self.env, con, "Exiting due to terminate flag being set to true",
                            'AI.MODELEXECUTE', 'inf_loop_model{1}', 'INPUTS', 4, 'outer_scope_input{1}', 'iterations{1}',
                            'loop_cond{1}', 'loop_input{1}', 'OUTPUTS', 2, 'outer_scope_output{1}', 'loop_output{1}',
                            error_msg_is_substr=True)

    def test_multiple_working_threads(self):
        con = get_connection(self.env, '{1}')

        # Load another onnx model that will be executed on the same threads that use the kill switch
        model_pb = load_file_content('mnist.onnx')
        sample_raw = load_file_content('one.raw')
        ret = con.execute_command('AI.MODELSTORE', 'mnist{1}', 'ONNX', DEVICE, 'BLOB', model_pb)
        self.env.assertEqual(ret, b'OK')
        con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)

        def run_parallel_onnx_sessions(con):
            ret = con.execute_command('AI.MODELEXECUTE', 'mnist{1}', 'INPUTS', 1, 'a{1}', 'OUTPUTS', 1, 'b{1}')
            self.env.assertEqual(ret, b'OK')
            check_error_message(self.env, con, "Exiting due to terminate flag being set to true",
                            'AI.MODELEXECUTE', 'inf_loop_model{1}', 'INPUTS', 4, 'outer_scope_input{1}', 'iterations{1}',
                            'loop_cond{1}', 'loop_input{1}', 'OUTPUTS', 2, 'outer_scope_output{1}', 'loop_output{1}',
                                error_msg_is_substr=True)
            ret = con.execute_command('AI.MODELEXECUTE', 'mnist{1}', 'INPUTS', 1, 'a{1}', 'OUTPUTS', 1, 'b{1}')
            self.env.assertEqual(ret, b'OK')
        run_test_multiproc(self.env, '{1}', 8, run_parallel_onnx_sessions)

    def test_multiple_devices(self):
        con = get_connection(self.env, '{1}')
        # CPU run queue is created from the start, so if we used a device different than CPU, we should
        # have maximum of 2*THREADS_PER_QUEUE run sessions, and otherwise we should have THREADS_PER_QUEUE.
        devices = {'CPU', DEVICE}
        backends_info = get_info_section(con, 'backends_info')
        self.env.assertEqual(backends_info['ai_onnxruntime_maximum_run_sessions_number'],
                             str(len(devices)*self.threads_per_queue))

        # Load another onnx model as if it runs on a different device (to test existence of multiple queues, and
        # the extension of the global onnx run sessions array as a consequence.)
        model_pb = load_file_content('mnist.onnx')
        ret = con.execute_command('AI.MODELSTORE', 'mnist_{1}', 'ONNX', 'CPU:1', 'BLOB', model_pb)
        self.env.assertEqual(ret, b'OK')
        devices.add('CPU:1')

        backends_info = get_info_section(con, 'backends_info')
        self.env.assertEqual(backends_info['ai_onnxruntime_maximum_run_sessions_number'],
                             str(len(devices)*self.threads_per_queue))


def test_forbidden_external_initializers(env):
    if not TEST_ONNX:
        env.debugPrint("skipping {} since TEST_ONNX=0".format(sys._getframe().f_code.co_name), force=True)
        return

    # The DISABLE_EXTERNAL_INITIALIZERS flag is used for CPU build only
    if DEVICE != 'CPU':
        return
    con = get_connection(env, '{1}')

    # move the external initializer to the redis' current dir (tests/flow/logs)
    external_initializer_model = load_file_content("model_with_external_initializers.onnx")
    shutil.copy(ROOT+"/tests/flow/test_data/Pads.bin", ROOT+"/tests/flow/logs")
    check_error_message(env, con, "Initializer tensors with external data is not allowed.",
                        'AI.MODELSTORE', 'ext_initializers_model{1}', 'ONNX', DEVICE,
                        'BLOB', external_initializer_model)

    os.remove(ROOT+"/tests/flow/logs/Pads.bin")
