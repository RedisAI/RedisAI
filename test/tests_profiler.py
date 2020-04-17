from RLTest.Profilers.stackUtil import fromFoldedStacksToDataframe
from includes import *

'''
python -m RLTest --test tests_common.py --module path/to/redisai.so
'''

PROFILER=0

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

    res = env.startProfiler(frequency=999)
    for datatype in tested_datatypes:
        for tensor_number in range(1, 10000):
            ret = con.execute_command('AI.TENSORSET', 'tensor_blob_{0}_{1}'.format(datatype, tensor_number), datatype,
                                      2, 'BLOB', tested_datatypes_blobs[datatype])
            env.assertEqual(ret, b'OK')
    stopandwrapUpProfiler(env, sys._getframe().f_code.co_name)


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

    res = env.startProfiler(frequency=999)
    for datatype in tested_datatypes:
        for tensor_number in range(1, 10000):
            ret = con.execute_command('AI.TENSORSET', 'tensor_blob_{0}_{1}'.format(datatype, tensor_number), datatype,
                                      1, 256, 'BLOB', tested_datatypes_blobs[datatype])
            env.assertEqual(ret, b'OK')
    stopandwrapUpProfiler(env, sys._getframe().f_code.co_name)


def test_profile_large_tensorset(env):
    if not PROFILER:
        env.debugPrint("skipping {} since PROFILER!=1".format(
            sys._getframe().f_code.co_name), force=True)
        return
    con = env.getConnection()

    model_pb, labels, img = load_mobilenet_test_data()
    res = env.startProfiler(frequency=999)
    for tensor_number in range(1, 10000):
        ret = con.execute_command('AI.TENSORSET', 'tensor_{0}'.format(tensor_number),
                                  'FLOAT', 1, img.shape[1], img.shape[0], img.shape[2],
                                  'BLOB', img.tobytes())
        env.assertEqual(ret, b'OK')

    stopandwrapUpProfiler(env, sys._getframe().f_code.co_name)


def test_profile_dagrun(env):
    # if not PROFILER:
    #     env.debugPrint("skipping {} since PROFILER!=1".format(
    #         sys._getframe().f_code.co_name), force=True)
    #     return
    con = env.getConnection()
    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    ret = con.execute_command('AI.MODELSET', 'financialNet', 'TF', "CPU",
                              'INPUTS', 'transaction', 'reference', 'OUTPUTS', 'output', model_pb)
    env.assertEqual(ret, b'OK')

    tensor_number = 1
    for reference_tensor in creditcard_referencedata:
        ret = con.execute_command(  'AI.TENSORSET', 'referenceTensor:{0}'.format(tensor_number),
                                  'FLOAT', 1, 256,
                                  'BLOB', reference_tensor.tobytes())
        env.assertEqual(ret, b'OK')
        tensor_number = tensor_number + 1

    tensor_number = 1
    res = env.startProfiler(frequency=999)
    for transaction_tensor in creditcard_transactions:
        for repetition in range(1,10):
            ret = con.execute_command(
                'AI.DAGRUN', 'LOAD', '1', 'referenceTensor:{}'.format(tensor_number), '|>',
                'AI.TENSORSET', 'transactionTensor:{}'.format(tensor_number), 'FLOAT', 1, 30,'BLOB', transaction_tensor.tobytes(), '|>',
                'AI.MODELRUN', 'financialNet', 
                            'INPUTS', 'transactionTensor:{}'.format(tensor_number), 'referenceTensor:{}'.format(tensor_number),
                            'OUTPUTS', 'classificationTensor:{}'.format(tensor_number), '|>',
                'AI.TENSORGET', 'classificationTensor:{}'.format(tensor_number), 'BLOB', 
            )
    stopandwrapUpProfiler(env, sys._getframe().f_code.co_name)

def stopandwrapUpProfiler(env,testname):
    res = env.stopProfiler()
    env.assertEqual(res, True)
    env.debugPrint("{0} perf.data file {1}".format(testname, env.getProfilerOutputs()),
                   force=True)
    res = env.generateTraceFiles()
    env.assertEqual(res, True)
    res = env.stackCollapse()
    env.assertEqual(res, True)
    stacks = env.getCollapsedStacksMap()
    for filename, stacksMap in stacks.items():
        env.debugPrint(
            "{0} collapsed stacks {1} len {2}".format(testname, filename, len(stacksMap)),
            force=True)
        df = fromFoldedStacksToDataframe(stacksMap, "RedisAI_TensorSet_RedisCommand", threshold=1)
        # render dataframe as html
        html = df.to_html(index=False)

        # write html to file
        text_file = open("{}.html".format(testname), "w")
        env.debugPrint(
            "{}.html".format(testname),
            force=True)
        text_file.write(html)
        text_file.close()