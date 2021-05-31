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

    con = env.getConnection()
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
                        'AI.MODELEXECUTE', 'm{1}', 'INPUTS', 3, 'a{1}', 'b{1}', 'c{1}', 'OTUPUTS', 'c{1}')

    con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 1, 'a{1}', 'OUTPUTS', 1, 'b{1}')

    ensureSlaveSynced(con, env)

    values = con.execute_command('AI.TENSORGET', 'b{1}', 'VALUES')
    argmax = max(range(len(values)), key=lambda i: values[i])
    env.assertEqual(argmax, 1)

    if env.useSlaves:
        con2 = env.getSlaveConnection()
        values2 = con2.execute_command('AI.TENSORGET', 'b{1}', 'VALUES')
        env.assertEqual(values2, values)


def test_onnx_modelrun_batchdim_mismatch(env):
    if not TEST_ONNX:
        env.debugPrint("skipping {} since TEST_ONNX=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()
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

    con = env.getConnection()
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
        con = env.getConnection()
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

    con = env.getConnection()

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

    con = env.getConnection()
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

    con = env.getConnection()
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


def test_onnx_model_rdb_save_load(env):
    env.skipOnCluster()
    if env.useAof or not TEST_ONNX:
        env.debugPrint("skipping {}".format(sys._getframe().f_code.co_name), force=True)
        return

    linear_model = load_file_content('linear_iris.onnx')

    con = env.getConnection()
    ret = con.execute_command('AI.MODELSTORE', 'linear{1}', 'ONNX', DEVICE, 'BLOB', linear_model)
    env.assertEqual(ret, b'OK')

    model_serialized_memory = con.execute_command('AI.MODELGET', 'linear{1}', 'BLOB')

    ensureSlaveSynced(con, env)
    ret = con.execute_command('SAVE')
    env.assertEqual(ret, True)

    env.stop()
    env.start()
    con = env.getConnection()
    model_serialized_after_rdbload = con.execute_command('AI.MODELGET', 'linear{1}', 'BLOB')
    env.assertEqual(len(model_serialized_memory), len(model_serialized_after_rdbload))
    env.assertEqual(len(linear_model), len(model_serialized_after_rdbload))
    # Assert in memory model binary is equal to loaded model binary
    env.assertTrue(model_serialized_memory == model_serialized_after_rdbload)
    # Assert input model binary is equal to loaded model binary
    env.assertTrue(linear_model == model_serialized_after_rdbload)


def tests_onnx_info(env):
    if not TEST_ONNX:
        env.debugPrint("skipping {} since TEST_ONNX=0".format(sys._getframe().f_code.co_name), force=True)
        return
    con = env.getConnection()

    ret = con.execute_command('AI.INFO')
    env.assertEqual(6, len(ret))

    linear_model = load_file_content('linear_iris.onnx')

    con.execute_command('AI.MODELSTORE', 'linear{1}', 'ONNX', DEVICE, 'BLOB', linear_model)
    
    ret = con.execute_command('AI.INFO')
    env.assertEqual(8, len(ret))
    env.assertEqual(b'ONNX version', ret[6])


def test_parallelism():
    env = Env(moduleArgs='INTRA_OP_PARALLELISM 1 INTER_OP_PARALLELISM 1')
    if not TEST_ONNX:
        env.debugPrint("skipping {} since TEST_ONNX=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()
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

    load_time_config = {k.split(":")[0]: k.split(":")[1]
                        for k in con.execute_command("INFO MODULES").decode().split("#")[3].split()[1:]}
    env.assertEqual(load_time_config["ai_inter_op_parallelism"], "1")
    env.assertEqual(load_time_config["ai_intra_op_parallelism"], "1")

    env = Env(moduleArgs='INTRA_OP_PARALLELISM 2 INTER_OP_PARALLELISM 2')
    load_time_config = {k.split(":")[0]: k.split(":")[1]
                        for k in con.execute_command("INFO MODULES").decode().split("#")[3].split()[1:]}
    env.assertEqual(load_time_config["ai_inter_op_parallelism"], "2")
    env.assertEqual(load_time_config["ai_intra_op_parallelism"], "2")


def test_onnx_use_custom_allocator(env):
    if not TEST_ONNX:
        env.debugPrint("skipping {} since TEST_ONNX=0".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()
    model_pb = load_file_content('mul_1.onnx')

    ai_memory_config = {k.split(":")[0]: k.split(":")[1]
                            for k in con.execute_command("INFO MODULES").decode().split("#")[4].split()[1:]}
    env.assertEqual(int(ai_memory_config["ai_onnxruntime_memory"]), 0)

    # Expect using the allocator during model set for allocating the model, its input name and output name:
    # overall 3 allocations. The model raw size is 130B ,and the names are 2B each. In practice we allocate
    # more than 134B as Redis allocator will use additional memory for its internal management and for the
    # 64-Byte alignment. When the test runs with valgrind, redis will use malloc for the allocations
    # (hence will not use additional memory).
    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'ONNX', 'CPU', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    ai_memory_config = {k.split(":")[0]: k.split(":")[1]
                        for k in con.execute_command("INFO MODULES").decode().split("#")[4].split()[1:]}

    # Expect using at least 130+63+(size of an address) + 2*(2+63+(size of an address)) bytes.
    model_allocation_bytes_used = int(ai_memory_config["ai_onnxruntime_memory"])
    env.assertTrue(model_allocation_bytes_used > 334)
    env.assertEqual(int(ai_memory_config["ai_onnxruntime_memory_access_num"]), 3)
    con.execute_command('AI.TENSORSET', 'a_mul{1}', 'FLOAT', 3, 2, 'VALUES', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    # Running the model should access the allocator 6 times: allocating+freeing input+output names,
    # and allocating+freeing the output as OrtValue.
    con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 1, 'a_mul{1}', 'OUTPUTS', 1, 'b{1}')
    values = con.execute_command('AI.TENSORGET', 'b{1}', 'VALUES')
    env.assertEqual(values, [b'1', b'4', b'9', b'16', b'25', b'36'])
    ai_memory_config = {k.split(":")[0]: k.split(":")[1]
                        for k in con.execute_command("INFO MODULES").decode().split("#")[4].split()[1:]}
    env.assertEqual(int(ai_memory_config["ai_onnxruntime_memory_access_num"]), 9)
    env.assertEqual(int(ai_memory_config["ai_onnxruntime_memory"]), model_allocation_bytes_used)

    # Expect using the allocator free function 3 times: when releasing the model, input name and output name.
    con.execute_command('AI.MODELDEL', 'm{1}')
    env.assertFalse(con.execute_command('EXISTS', 'm{1}'))
    ai_memory_config = {k.split(":")[0]: k.split(":")[1]
                        for k in con.execute_command("INFO MODULES").decode().split("#")[4].split()[1:]}
    env.assertEqual(int(ai_memory_config["ai_onnxruntime_memory"]), 0)
    env.assertEqual(int(ai_memory_config["ai_onnxruntime_memory_access_num"]), 12)

    # test the use of Redis allocator in model run op.
    model_pb = load_file_content('mnist.onnx')
    sample_raw = load_file_content('one.raw')

    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'ONNX', 'CPU', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)

    # Expect 18 allocator's access from onnx during the run (in addition to the allocations that were made while
    # creating the model).
    ai_memory_config = {k.split(":")[0]: k.split(":")[1]
                        for k in con.execute_command("INFO MODULES").decode().split("#")[4].split()[1:]}
    allocator_access_num_before = ai_memory_config["ai_onnxruntime_memory_access_num"]
    con.execute_command('AI.MODELEXECUTE', 'm{1}', 'INPUTS', 1, 'a{1}', 'OUTPUTS', 1, 'b{1}')
    ai_memory_config = {k.split(":")[0]: k.split(":")[1]
                        for k in con.execute_command("INFO MODULES").decode().split("#")[4].split()[1:]}
    allocator_access_num_after = ai_memory_config["ai_onnxruntime_memory_access_num"]
    env.assertEqual(int(allocator_access_num_after) - int(allocator_access_num_before), 18)

    values = con.execute_command('AI.TENSORGET', 'b{1}', 'VALUES')
    argmax = max(range(len(values)), key=lambda i: values[i])
    env.assertEqual(argmax, 1)


def test_onnx_use_custom_allocator_with_GPU(env):
    if not TEST_ONNX:
        env.debugPrint("skipping {} since TEST_ONNX=0".format(sys._getframe().f_code.co_name), force=True)
        return
    if DEVICE == 'CPU':
        env.debugPrint("skipping {} since this test if for GPU only".format(sys._getframe().f_code.co_name), force=True)
        return

    con = env.getConnection()
    model_pb = load_file_content('mul_1.onnx')
    ai_memory_config = {k.split(":")[0]: k.split(":")[1]
                        for k in con.execute_command("INFO MODULES").decode().split("#")[4].split()[1:]}
    env.assertEqual(int(ai_memory_config["ai_onnxruntime_memory"]), 0)

    # Expect using the allocator during model set for allocating the model, its input name and output name:
    # overall 3 allocations. The model raw size is 130B ,and the names are 2B each. In practice we allocate
    # more than 134B as Redis allocator will use additional memory for its internal management and for the
    # 64-Byte alignment. When the test runs with valgrind, redis will use malloc for the allocations.
    ret = con.execute_command('AI.MODELSTORE', 'm_gpu{1}', 'ONNX', DEVICE, 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    # but for GPU, expect using the allocator only for allocating input and output names (not the model itself).
    ret = con.execute_command('AI.MODELSTORE', 'm_cpu{1}', 'ONNX', 'CPU', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    ai_memory_config = {k.split(":")[0]: k.split(":")[1]
                        for k in con.execute_command("INFO MODULES").decode().split("#")[4].split()[1:]}

    # Expect using at least 130+63+(size of an address) + 4*(2+63+(size of an address)) bytes.
    model_allocation_bytes_used = int(ai_memory_config["ai_onnxruntime_memory"])
    env.assertTrue(model_allocation_bytes_used > 472)
    env.assertTrue(model_allocation_bytes_used < 705)
    env.assertEqual(int(ai_memory_config["ai_onnxruntime_memory_access_num"]), 5)

    # Make sure that allocator is not used for running and freeing the GPU model, except for
    # the input and output names allocations (and deallocations).
    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 3, 2, 'VALUES', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    con.execute_command('AI.MODELEXECUTE', 'm_gpu{1}', 'INPUTS', 1, 'a{1}', 'OUTPUTS', 1, 'b{1}')
    values = con.execute_command('AI.TENSORGET', 'b{1}', 'VALUES')
    env.assertEqual(values, [b'1', b'4', b'9', b'16', b'25', b'36'])
    # Expect that memory usage didn't change, and for another 4 accesses to the allocator (input and output names
    # allocation and free)
    ai_memory_config = {k.split(":")[0]: k.split(":")[1]
                        for k in con.execute_command("INFO MODULES").decode().split("#")[4].split()[1:]}
    env.assertEqual(int(ai_memory_config["ai_onnxruntime_memory"]), model_allocation_bytes_used)
    env.assertEqual(int(ai_memory_config["ai_onnxruntime_memory_access_num"]), 9)

    # Expect only 2 more accesses in delete - for deallocating input and output names
    con.execute_command('AI.MODELDEL', 'm_gpu{1}')
    env.assertFalse(con.execute_command('EXISTS', 'm_gpu{1}'))
    ai_memory_config = {k.split(":")[0]: k.split(":")[1]
                        for k in con.execute_command("INFO MODULES").decode().split("#")[4].split()[1:]}
    env.assertEqual(int(ai_memory_config["ai_onnxruntime_memory_access_num"]), 11)


def test_onnx_kill_switch(env):
    con = env.getConnection()
    model_with_inf_loop = load_file_content("model_with_infinite_loop.onnx")
    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'ONNX', DEVICE, 'BLOB', model_with_inf_loop)
    env.assertEqual(ret, b'OK')
    model = con.execute_command('AI.MODELGET', 'm{1}', 'META', 'BLOB')
    env.debugPrint(str(model), force=True)

    # Set tensors according to the model inputs

    ret = con.execute_command('AI.TENSORSET', 'in{1}', 'int64', DEVICE, 'BLOB', model_with_inf_loop)
