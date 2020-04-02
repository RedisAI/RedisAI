from includes import *

'''
python -m RLTest --test tests_common.py --module path/to/redisai.so
'''


def test_profile_small_tensorset(env):
    if not PROFILER:
        env.debugPrint("skipping {} since PROFILER!=1".format(
            sys._getframe().f_code.co_name), force=True)
        return
    con = env.getConnection()

    tested_datatypes = ["FLOAT", "DOUBLE", "INT8", "INT16", "INT32", "INT64", "UINT8", "UINT16"]
    tested_datatypes_blobs = {}

    for datatype in tested_datatypes:
        ret = con.execute_command('AI.TENSORSET', 'tensor_{0}'.format(datatype), datatype, 2, 'VALUES', 1, 1)
        env.assertEqual(ret, b'OK')

    # AI.TENSORGET in BLOB format and set in a new key
    for datatype in tested_datatypes:
        tensor_dtype, tensor_dim, tensor_blob = con.execute_command('AI.TENSORGET', 'tensor_{0}'.format(datatype),
                                                                    'BLOB')
        tested_datatypes_blobs[datatype] = tensor_blob

    res = env.startProfiler(999)
    for datatype in tested_datatypes:
        for tensor_number in range(1, 10000):
            ret = con.execute_command('AI.TENSORSET', 'tensor_blob_{0}_{1}'.format(datatype, tensor_number), datatype,
                                      2, 'BLOB', tested_datatypes_blobs[datatype])
            env.assertEqual(ret, b'OK')
    res = env.stopProfiler()
    env.debugPrint("{0} perf.data file {1}".format(sys._getframe().f_code.co_name, env.getProfilerOutputs()),
                   force=True)


def test_profile_medium_tensorset(env):
    if not PROFILER:
        env.debugPrint("skipping {} since PROFILER!=1".format(
            sys._getframe().f_code.co_name), force=True)
        return
    con = env.getConnection()

    tested_datatypes = ["FLOAT", "DOUBLE", "INT8", "INT16", "INT32", "INT64", "UINT8", "UINT16"]
    tested_datatypes_blobs = {}

    for datatype in tested_datatypes:
        ret = con.execute_command('AI.TENSORSET', 'tensor_{0}'.format(datatype), datatype, 1, 256)
        env.assertEqual(ret, b'OK')

    # AI.TENSORGET in BLOB format and set in a new key
    for datatype in tested_datatypes:
        tensor_dtype, tensor_dim, tensor_blob = con.execute_command('AI.TENSORGET', 'tensor_{0}'.format(datatype),
                                                                    'BLOB')
        tested_datatypes_blobs[datatype] = tensor_blob

    res = env.startProfiler(999)
    for datatype in tested_datatypes:
        for tensor_number in range(1, 10000):
            ret = con.execute_command('AI.TENSORSET', 'tensor_blob_{0}_{1}'.format(datatype, tensor_number), datatype,
                                      1, 256, 'BLOB', tested_datatypes_blobs[datatype])
            env.assertEqual(ret, b'OK')
    res = env.stopProfiler()
    env.debugPrint("{0} perf.data file {1}".format(sys._getframe().f_code.co_name, env.getProfilerOutputs()),
                   force=True)


def test_profile_large_tensorset(env):
    if not PROFILER:
        env.debugPrint("skipping {} since PROFILER!=1".format(
            sys._getframe().f_code.co_name), force=True)
        return
    con = env.getConnection()

    model_pb, labels, img = load_mobilenet_test_data()
    res = env.startProfiler(999)
    for tensor_number in range(1, 10000):
        ret = con.execute_command('AI.TENSORSET', 'tensor_{0}'.format(tensor_number),
                                  'FLOAT', 1, img.shape[1], img.shape[0], img.shape[2],
                                  'BLOB', img.tobytes())
        env.assertEqual(ret, b'OK')
    res = env.stopProfiler()
    env.debugPrint("{0} perf.data file {1}".format(sys._getframe().f_code.co_name, env.getProfilerOutputs()),
                   force=True)
