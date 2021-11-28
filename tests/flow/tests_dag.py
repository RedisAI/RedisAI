import redis
from functools import wraps
import multiprocessing as mp
from includes import *
import time 

'''
python -m RLTest --test tests_dag.py --module path/to/redisai.so
'''

# change this to make inference tests longer
MAX_TRANSACTIONS=100


def test_dagrun_modelexecute_scriptexecute_resnet(env):
    if (not TEST_TF or not TEST_PT):
        return
    if(VALGRIND):
        env.debugPrint("skipping {} since it's hanging CI".format(sys._getframe().f_code.co_name), force=True)
        env.skip()
    con = get_connection(env, '{1}')
    model_name = 'imagenet_model:{1}'
    script_name = 'imagenet_script:{1}'
    image_key = 'image:{1}'
    temp_key1 = 'temp_key1'
    temp_key2 = 'temp_key2'
    class_key = 'output:{1}'
    inputvar = 'images'
    outputvar = 'output'
    model_pb, script, labels, img = load_resnet_test_data()

    ret = con.execute_command('AI.MODELSTORE', model_name, 'TF', DEVICE,
                              'INPUTS', 1, inputvar,
                              'OUTPUTS', 1, outputvar,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.SCRIPTSTORE', script_name, DEVICE, 'ENTRY_POINTS', 4, 'pre_process_3ch', 'pre_process_4ch', 'post_process', 'ensemble', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    for opnumber in range(1,100):
        ret = con.execute_command(
            'AI.DAGEXECUTE', 'PERSIST', '1', class_key, '|>',
            'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(), '|>',
            'AI.SCRIPTEXECUTE',  script_name, 'pre_process_3ch',
                         'INPUTS', 1, image_key,
                         'OUTPUTS', 1, temp_key1,  '|>',
            'AI.MODELEXECUTE', model_name,
                         'INPUTS', 1, temp_key1,
                         'OUTPUTS', 1, temp_key2,  '|>',
            'AI.SCRIPTEXECUTE',  script_name, 'post_process',
                          'INPUTS', 1, temp_key2,
                          'OUTPUTS', 1, class_key
        )
        env.assertEqual([b'OK',b'OK',b'OK',b'OK'],ret)

        ret = con.execute_command('AI.TENSORGET', class_key, 'VALUES' )
        # tf model has 100 classes [0,999]
        env.assertEqual(ret[0]>=0 and ret[0]<1001, True)


def test_dag_modelexecute_financialNet_separate_tensorget(env):
    if not TEST_TF:
        return
    con = get_connection(env, '{1}')

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    model_name = 'financialNet{1}'

    ret = con.execute_command('AI.MODELSTORE', model_name, 'TF', "CPU",
                              'INPUTS', 2, 'transaction', 'reference', 'OUTPUTS', 1, 'output', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    for tensor_number in range(1,MAX_TRANSACTIONS):
        for repetition in range(1,10):
            reference_tensor = creditcard_referencedata[tensor_number]
            transaction_tensor = creditcard_transactions[tensor_number]
            result_tensor_keyname = 'resultTensor{{1}}{}'.format(tensor_number)
            reference_tensor_keyname = 'referenceTensor{{1}}{}'.format(tensor_number)
            transaction_tensor_keyname = 'transactionTensor{{1}}{}'.format(tensor_number)
            
            ret = con.execute_command('AI.TENSORSET', reference_tensor_keyname,
                                    'FLOAT', 1, 256,
                                    'BLOB', reference_tensor.tobytes())
            env.assertEqual(ret, b'OK')
            ret = con.execute_command("EXISTS {}".format(reference_tensor_keyname))
            env.assertEqual(ret, 1)

            ret = con.execute_command(
                'AI.DAGEXECUTE', 'LOAD', '1', reference_tensor_keyname,
                'PERSIST', '1', result_tensor_keyname, '|>',
                'AI.TENSORSET', transaction_tensor_keyname, 'FLOAT', 1, 30,'BLOB', transaction_tensor.tobytes(), '|>',
                'AI.MODELEXECUTE', model_name,
                    'INPUTS', 2, transaction_tensor_keyname, reference_tensor_keyname,
                    'OUTPUTS', 1, result_tensor_keyname,
            )
            env.assertEqual([b'OK',b'OK'],ret)

            ret = con.execute_command("AI.TENSORGET {} META".format(
                result_tensor_keyname))
            env.assertEqual([b'dtype', b'FLOAT', b'shape', [1, 2]], ret)


def test_dag_modelexecute_financialNet(env):
    if not TEST_TF:
        return
    con = get_connection(env, '{1}')

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    model_name = 'financialNet{1}'

    ret = con.execute_command('AI.MODELSTORE', model_name, 'TF', "CPU",
                              'INPUTS', 2, 'transaction', 'reference', 'OUTPUTS', 1, 'output', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    for tensor_number in range(1,MAX_TRANSACTIONS):
        for repetition in range(1,10):
            reference_tensor = creditcard_referencedata[tensor_number]
            transaction_tensor = creditcard_transactions[tensor_number]
            result_tensor_keyname = 'resultTensor{{1}}{}'.format(tensor_number)
            reference_tensor_keyname = 'referenceTensor{{1}}{}'.format(tensor_number)
            transaction_tensor_keyname = 'transactionTensor{{1}}{}'.format(tensor_number)

            ret = con.execute_command('AI.TENSORSET', reference_tensor_keyname,
                                    'FLOAT', 1, 256,
                                    'BLOB', reference_tensor.tobytes())
            env.assertEqual(ret, b'OK')
            ret = con.execute_command("EXISTS {}".format(reference_tensor_keyname))
            env.assertEqual(ret, 1)

            ret = con.execute_command(
                'AI.DAGEXECUTE', 'LOAD', '1', reference_tensor_keyname,
                            'PERSIST', '1', result_tensor_keyname, '|>',
                'AI.TENSORSET', transaction_tensor_keyname, 'FLOAT', 1, 30,'BLOB', transaction_tensor.tobytes(), '|>',
                'AI.MODELEXECUTE', model_name,
                            'INPUTS', 2, transaction_tensor_keyname, reference_tensor_keyname,
                            'OUTPUTS', 1, result_tensor_keyname, '|>',
                'AI.TENSORGET', result_tensor_keyname, 'META',
            )
            env.assertEqual([b'OK',b'OK',[b'dtype', b'FLOAT', b'shape', [1, 2]]], ret)

            # assert that transaction tensor does not exist
            ret = con.execute_command("EXISTS {}".format(transaction_tensor_keyname))
            env.assertEqual(ret, 0)
            # assert that result tensor exists
            ret = con.execute_command("EXISTS {}".format(result_tensor_keyname))
            env.assertEqual(ret, 1)


def test_dag_modelexecute_financialNet_autobatch(env):
    if not TEST_TF:
        return
    con = get_connection(env, '{1}')

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    model_name = 'financialNet{1}'

    ret = con.execute_command('AI.MODELSTORE', model_name, 'TF', 'CPU',
                              'BATCHSIZE', 2, 'MINBATCHSIZE', 2,
                              'INPUTS', 2, 'transaction', 'reference', 'OUTPUTS', 1, 'output',
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    for tensor_number in range(1,MAX_TRANSACTIONS):
        for repetition in range(1,10):
            reference_tensor = creditcard_referencedata[tensor_number]
            transaction_tensor = creditcard_transactions[tensor_number]
            result_tensor_keyname = 'resultTensor{{1}}{}'.format(tensor_number)
            reference_tensor_keyname = 'referenceTensor{{1}}{}'.format(tensor_number)
            transaction_tensor_keyname = 'transactionTensor{{1}}{}'.format(tensor_number)

            ret = con.execute_command('AI.TENSORSET', reference_tensor_keyname,
                                    'FLOAT', 1, 256,
                                    'BLOB', reference_tensor.tobytes())
            env.assertEqual(ret, b'OK')
            ret = con.execute_command("EXISTS {}".format(reference_tensor_keyname))
            env.assertEqual(ret, 1)

            def run():
                con = get_connection(env, '{1}')
                ret = con.execute_command(
                    'AI.DAGEXECUTE', 'LOAD', '1', reference_tensor_keyname, '|>',
                    'AI.TENSORSET', transaction_tensor_keyname, 'FLOAT', 1, 30,'BLOB', transaction_tensor.tobytes(), '|>',
                    'AI.MODELEXECUTE', model_name,
                                'INPUTS', 2, transaction_tensor_keyname, reference_tensor_keyname,
                                'OUTPUTS', 1, result_tensor_keyname
                )
                ensureSlaveSynced(con, env)

            t = threading.Thread(target=run)
            t.start()

            ret = con.execute_command(
                'AI.DAGEXECUTE', 'LOAD', '1', reference_tensor_keyname,
                            'PERSIST', '1', result_tensor_keyname, '|>',
                'AI.TENSORSET', transaction_tensor_keyname, 'FLOAT', 1, 30,'BLOB', transaction_tensor.tobytes(), '|>',
                'AI.MODELEXECUTE', model_name,
                            'INPUTS', 2, transaction_tensor_keyname, reference_tensor_keyname,
                            'OUTPUTS', 1, result_tensor_keyname, '|>',
                'AI.TENSORGET', result_tensor_keyname, 'META',
            )

            t.join()
            ensureSlaveSynced(con, env)

            env.assertEqual([b'OK',b'OK',[b'dtype', b'FLOAT', b'shape', [1, 2]]], ret)

            # assert that transaction tensor does not exist
            ret = con.execute_command("EXISTS {}".format(transaction_tensor_keyname))
            env.assertEqual(ret, 0)
            # assert that result tensor exists
            ret = con.execute_command("EXISTS {}".format(result_tensor_keyname))
            env.assertEqual(ret, 1)


def test_slowlog_time_dag_modelexecute_financialNet_autobatch(env):
    if not TEST_TF:
        return
    con = get_connection(env, '{1}')

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    model_name = 'financialNet{1}'

    ret = con.execute_command('AI.MODELSTORE', model_name, 'TF', 'CPU',
                              'BATCHSIZE', 2, 'MINBATCHSIZE', 2,
                              'INPUTS', 2, 'transaction', 'reference', 'OUTPUTS', 1, 'output',
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('CONFIG', 'SET', 'slowlog-log-slower-than', '1')
    env.assertEqual(ret, b'OK')
    total_time = 0
    abs_time = 0

    for tensor_number in range(1,MAX_TRANSACTIONS):
        for repetition in range(1,10):
            reference_tensor = creditcard_referencedata[tensor_number]
            transaction_tensor = creditcard_transactions[tensor_number]
            result_tensor_keyname = 'resultTensor{{1}}{}'.format(tensor_number)
            reference_tensor_keyname = 'referenceTensor{{1}}{}'.format(tensor_number)
            transaction_tensor_keyname = 'transactionTensor{{1}}{}'.format(tensor_number)

            ret = con.execute_command('AI.TENSORSET', reference_tensor_keyname,
                                    'FLOAT', 1, 256,
                                    'BLOB', reference_tensor.tobytes())
            env.assertEqual(ret, b'OK')
            ret = con.execute_command("EXISTS {}".format(reference_tensor_keyname))
            env.assertEqual(ret, 1)

            def run():
                con = get_connection(env, '{1}')
                ret = con.execute_command(
                    'AI.DAGEXECUTE', 'LOAD', '1', reference_tensor_keyname, '|>',
                    'AI.TENSORSET', transaction_tensor_keyname, 'FLOAT', 1, 30,'BLOB', transaction_tensor.tobytes(), '|>',
                    'AI.MODELEXECUTE', model_name,
                                'INPUTS', 2, transaction_tensor_keyname, reference_tensor_keyname,
                                'OUTPUTS', 1, result_tensor_keyname
                )
                ensureSlaveSynced(con, env)

            t = threading.Thread(target=run)

            start = time.time_ns()
            t.start()

            ret = con.execute_command(
                'AI.DAGEXECUTE', 'LOAD', '1', reference_tensor_keyname,
                            'PERSIST', '1', result_tensor_keyname, '|>',
                'AI.TENSORSET', transaction_tensor_keyname, 'FLOAT', 1, 30,'BLOB', transaction_tensor.tobytes(), '|>',
                'AI.MODELEXECUTE', model_name,
                            'INPUTS', 2, transaction_tensor_keyname, reference_tensor_keyname,
                            'OUTPUTS', 1, result_tensor_keyname, '|>',
                'AI.TENSORGET', result_tensor_keyname, 'META',
            )

            t.join()
            ensureSlaveSynced(con, env)
            end = time.time_ns()
            abs_time += (end - start)//1000
            curr_time = con.execute_command('SLOWLOG', 'GET', 1)[0][2]
            total_time += curr_time
            env.assertTrue((end - start)//1000 >= curr_time)

    info = con.execute_command('AI.INFO', model_name)
    financialNetRunInfo = info_to_dict(info)
    env.assertTrue(0 < financialNetRunInfo['duration'] <= total_time <= abs_time)


def test_slowlog_time_dag_modelexecute_financialNet_no_writes(env):
    if not TEST_TF:
        return
    con = get_connection(env, '{1}')

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    model_name = 'financialNet{1}'

    ret = con.execute_command('AI.MODELSTORE', model_name, 'TF', "CPU",
                              'INPUTS', 2, 'transaction', 'reference', 'OUTPUTS', 1,'output', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('CONFIG', 'SET', 'slowlog-log-slower-than', '1')
    env.assertEqual(ret, b'OK')
    total_time = 0
    abs_time = 0

    for tensor_number in range(1,MAX_TRANSACTIONS):
        for repetition in range(1,10):
            reference_tensor = creditcard_referencedata[tensor_number]
            transaction_tensor = creditcard_transactions[tensor_number]
            result_tensor_keyname = 'resultTensor{{1}}{}'.format(tensor_number)
            reference_tensor_keyname = 'referenceTensor{{1}}{}'.format(tensor_number)
            transaction_tensor_keyname = 'transactionTensor{{1}}{}'.format(tensor_number)
            
            ret = con.execute_command('AI.TENSORSET', reference_tensor_keyname,
                                    'FLOAT', 1, 256,
                                    'BLOB', reference_tensor.tobytes())
            env.assertEqual(ret, b'OK')
            ret = con.execute_command("EXISTS {}".format(reference_tensor_keyname))
            env.assertEqual(ret, 1)

            start = time.time_ns()
            ret = con.execute_command(
                'AI.DAGEXECUTE', 'LOAD', '1', reference_tensor_keyname, '|>',
                'AI.TENSORSET', transaction_tensor_keyname, 'FLOAT', 1, 30,'BLOB', transaction_tensor.tobytes(), '|>',
                'AI.MODELEXECUTE', model_name,
                            'INPUTS', 2, transaction_tensor_keyname, reference_tensor_keyname,
                            'OUTPUTS', 1, result_tensor_keyname, '|>',
                'AI.TENSORGET',result_tensor_keyname, 'META',  '|>',
                'AI.TENSORGET', result_tensor_keyname, 'VALUES'
            )
            end = time.time_ns()
            abs_time += (end - start)//1000
            curr_time = con.execute_command('SLOWLOG', 'GET', 1)[0][2]
            total_time += curr_time
            env.assertTrue((end - start)//1000 >= curr_time)

    info = con.execute_command('AI.INFO', model_name)
    financialNetRunInfo = info_to_dict(info)
    env.assertTrue(0 < financialNetRunInfo['duration'] <= total_time <= abs_time)


def test_dag_modelexecute_financialNet_no_writes(env):
    if not TEST_TF:
        return
    con = get_connection(env, '{1}')

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    model_name = 'financialNet{1}'

    ret = con.execute_command('AI.MODELSTORE', model_name, 'TF', "CPU",
                              'INPUTS', 2, 'transaction', 'reference', 'OUTPUTS', 1,'output', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    for tensor_number in range(1,MAX_TRANSACTIONS):
        for repetition in range(1,10):
            reference_tensor = creditcard_referencedata[tensor_number]
            transaction_tensor = creditcard_transactions[tensor_number]
            result_tensor_keyname = 'resultTensor{{1}}{}'.format(tensor_number)
            reference_tensor_keyname = 'referenceTensor{{1}}{}'.format(tensor_number)
            transaction_tensor_keyname = 'transactionTensor{{1}}{}'.format(tensor_number)
            
            ret = con.execute_command('AI.TENSORSET', reference_tensor_keyname,
                                    'FLOAT', 1, 256,
                                    'BLOB', reference_tensor.tobytes())
            env.assertEqual(ret, b'OK')
            ret = con.execute_command("EXISTS {}".format(reference_tensor_keyname))
            env.assertEqual(ret, 1)

            ret = con.execute_command(
                'AI.DAGEXECUTE', 'LOAD', '1', reference_tensor_keyname, '|>',
                'AI.TENSORSET', transaction_tensor_keyname, 'FLOAT', 1, 30,'BLOB', transaction_tensor.tobytes(), '|>',
                'AI.MODELEXECUTE', model_name,
                            'INPUTS', 2, transaction_tensor_keyname, reference_tensor_keyname,
                            'OUTPUTS', 1, result_tensor_keyname, '|>',
                'AI.TENSORGET',result_tensor_keyname, 'META',  '|>',
                'AI.TENSORGET', result_tensor_keyname, 'VALUES'
            )
            env.assertEqual(4, len(ret))
            env.assertEqual([b'OK', b'OK'], ret[:2])
            env.assertEqual([b'dtype', b'FLOAT', b'shape', [1, 2]], ret[2])
            values = ret[3]
            # Assert that resulting classification is within [0,1]
            env.assertEqual(True, 0 <= float(values[0]) <= 1)
            env.assertEqual(True, 0 <= float(values[1]) <= 1)

            # assert that transaction tensor does not exist
            ret = con.execute_command("EXISTS {}".format(transaction_tensor_keyname))
            env.assertEqual(ret, 0)
            # assert that result tensor exists
            ret = con.execute_command("EXISTS {}".format(result_tensor_keyname))
            env.assertEqual(ret, 0)


def test_dagro_modelexecute_financialNet_no_writes_multiple_modelruns(env):
    if not TEST_TF:
        return
    con = get_connection(env, '{1}')

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    model_name = 'financialNet_no_writes{1}'

    ret = con.execute_command('AI.MODELSTORE', model_name, 'TF', "CPU",
                              'INPUTS', 2, 'transaction', 'reference', 'OUTPUTS', 1, 'output', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    for tensor_number in range(1,MAX_TRANSACTIONS+1):
        for repetition in range(1,11):
            reference_tensor = creditcard_referencedata[tensor_number-1]
            transaction_tensor = creditcard_transactions[tensor_number-1]
            result_tensor_keyname = 'resultTensor{{1}}{}'.format(tensor_number)
            reference_tensor_keyname = 'referenceTensor{{1}}{}'.format(tensor_number)
            transaction_tensor_keyname = 'transactionTensor{{1}}{}'.format(tensor_number)
            
            ret = con.execute_command('AI.TENSORSET', reference_tensor_keyname,
                                    'FLOAT', 1, 256,
                                    'BLOB', reference_tensor.tobytes())
            env.assertEqual(ret, b'OK')
            ret = con.execute_command("EXISTS {}".format(reference_tensor_keyname))
            env.assertEqual(ret, 1)
            ret = con.execute_command(
                'AI.DAGEXECUTE_RO', 'LOAD', '1', reference_tensor_keyname, '|>',
                'AI.TENSORSET', transaction_tensor_keyname, 'FLOAT', 1, 30,'BLOB', transaction_tensor.tobytes(), '|>',
                'AI.MODELEXECUTE', model_name,
                            'INPUTS', 2, transaction_tensor_keyname, reference_tensor_keyname,
                            'OUTPUTS', 1, result_tensor_keyname, '|>',
                'AI.TENSORGET', result_tensor_keyname, 'META', 'VALUES', '|>',
                'AI.MODELEXECUTE', model_name,
                            'INPUTS', 2, transaction_tensor_keyname, reference_tensor_keyname,
                            'OUTPUTS', 1, result_tensor_keyname, '|>',
                'AI.TENSORGET', result_tensor_keyname, 'META', 'VALUES', 
            )
            env.assertEqual(5, len(ret))
            env.assertEqual([b'OK', b'OK'], ret[:2])
            env.assertEqual([b'dtype', b'FLOAT', b'shape', [1, 2]], ret[2][:4])
            env.assertEqual(b'OK', ret[3])
            env.assertEqual([b'dtype', b'FLOAT', b'shape', [1, 2]], ret[4][:4])
            for _, dtype, _, shape, _, values in [ret[2], ret[4]]:
                # Assert that resulting classification is within [0,1]
                env.assertEqual(True, 0 <= float(values[0]) <= 1)
                env.assertEqual(True, 0 <= float(values[1]) <= 1)

    info = con.execute_command('AI.INFO', model_name)
    financialNetRunInfo = info_to_dict(info)

    env.assertEqual(model_name, financialNetRunInfo['key'])
    env.assertEqual('MODEL', financialNetRunInfo['type'])
    env.assertEqual('TF', financialNetRunInfo['backend'])
    # Commenting due to: 'ascii' codec can't encode character '\u274c' in position 8: ordinal not in range(128)
    # env.assertEqual(DEVICE, financialNetRunInfo['device']) 
    env.assertTrue(financialNetRunInfo['duration'] > 0)
    env.assertEqual(2*MAX_TRANSACTIONS*10, financialNetRunInfo['samples'])
    env.assertEqual(2*MAX_TRANSACTIONS*10, financialNetRunInfo['calls'])
    env.assertEqual(0, financialNetRunInfo['errors'])

    con.execute_command('AI.INFO', model_name, 'RESETSTAT')
    info = con.execute_command('AI.INFO', model_name)
    financialNetRunInfo = info_to_dict(info)

    env.assertEqual(model_name, financialNetRunInfo['key'])
    env.assertEqual('MODEL', financialNetRunInfo['type'])
    env.assertEqual('TF', financialNetRunInfo['backend'])
    # Commenting due to: 'ascii' codec can't encode character '\u274c' in position 8: ordinal not in range(128)
    # env.assertEqual(DEVICE, financialNetRunInfo['device'])
    env.assertEqual(0, financialNetRunInfo['duration'])
    env.assertEqual(0, financialNetRunInfo['samples'])
    env.assertEqual(0, financialNetRunInfo['calls'])
    env.assertEqual(0, financialNetRunInfo['errors'])


def test_dagexecute_modelexecute_multidevice_resnet(env):
    if (not TEST_TF or not TEST_PT):
        return
    con = get_connection(env, '{1}')
    model_name_0 = 'imagenet_model1:{1}'
    model_name_1 = 'imagenet_model2:{1}'
    script_name = 'imagenet_script:{1}'
    image_key = 'image:{1}'
    temp_key1 = 'temp_key1:{1}'
    temp_key2_0 = 'temp_key2_0'
    temp_key2_1 = 'temp_key2_1'
    class_key_0 = 'output0:{1}'
    class_key_1 = 'output1:{1}'
    inputvar = 'images'
    outputvar = 'output'
    model_pb, script, labels, img = load_resnet_test_data()

    device_0 = 'CPU:1'
    device_1 = DEVICE

    ret = con.execute_command('AI.MODELSTORE', model_name_0, 'TF', device_0,
                              'INPUTS', 1, inputvar,
                              'OUTPUTS', 1, outputvar,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELSTORE', model_name_1, 'TF', device_1,
                              'INPUTS', 1, inputvar,
                              'OUTPUTS', 1, outputvar,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.SCRIPTSTORE', script_name, device_0, 'ENTRY_POINTS', 4, 'pre_process_3ch', 'pre_process_4ch', 'post_process', 'ensemble', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    check_error_message(env, con, "INPUT key cannot be found in DAG",
                        'AI.DAGEXECUTE', 'ROUTING', image_key, '|>', 'AI.SCRIPTEXECUTE',  script_name, 'pre_process_3ch',
                        'INPUTS', 1, image_key, 'OUTPUTS', 1, temp_key1)

    check_error_message(env, con, "INPUT key cannot be found in DAG",
                        'AI.DAGEXECUTE', 'ROUTING',  image_key, '|>', 'AI.MODELEXECUTE', model_name_0,
                        'INPUTS', 1, image_key, 'OUTPUTS', 1, temp_key1)

    ret = con.execute_command('AI.DAGEXECUTE', 
                            'ROUTING', '{1}','|>',
                            'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(),'|>',
                            'AI.SCRIPTEXECUTE',  script_name, 'wrong_fn',
                            'INPUTS', 1, image_key,
                            'OUTPUTS', 1, temp_key1)
    env.assertEqual(b'OK', ret[0])
    env.assertEqual(type(ret[1]), redis.exceptions.ResponseError)
    env.assertEqual("Function does not exist: wrong_fn",  ret[1].__str__())

    check_error_message(env, con, "Number of keys given as INPUTS here does not match model definition",
                        'AI.DAGEXECUTE', 'ROUTING', '{1}',
                        '|>', 'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(),
                        '|>',
                        'AI.SCRIPTEXECUTE',  script_name, 'pre_process_3ch',
                        'INPUTS', 1, image_key,
                        'OUTPUTS', 1, temp_key1, '|>',
                        'AI.MODELEXECUTE', model_name_0,
                        'INPUTS', 2, temp_key1, temp_key1,
                        'OUTPUTS', 1, temp_key2_0)

    ret = con.execute_command(
        'AI.DAGEXECUTE',
                     'PERSIST', '2', class_key_0, class_key_1, '|>',
        'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(),
                     '|>',
        'AI.SCRIPTEXECUTE', script_name, 'pre_process_3ch',
                     'INPUTS', 1, image_key,
                     'OUTPUTS', 1, temp_key1,
                     '|>',
        'AI.MODELEXECUTE', model_name_0,
                     'INPUTS', 1, temp_key1,
                     'OUTPUTS', 1, temp_key2_0,
                     '|>',
        'AI.MODELEXECUTE', model_name_1,
                     'INPUTS', 1, temp_key1,
                     'OUTPUTS', 1, temp_key2_1,
                     '|>',
        'AI.SCRIPTEXECUTE', script_name, 'post_process',
                      'INPUTS', 1, temp_key2_0,
                      'OUTPUTS', 1, class_key_0,
                      '|>',
        'AI.SCRIPTEXECUTE', script_name, 'post_process',
                      'INPUTS', 1, temp_key2_1,
                      'OUTPUTS', 1, class_key_1
    )
    env.assertEqual([b'OK', b'OK', b'OK', b'OK', b'OK', b'OK'], ret)

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.TENSORGET', class_key_0, 'VALUES' )
    # tf model has 100 classes [0,999]
    env.assertEqual(ret[0]>=0 and ret[0]<1001, True)

    ret = con.execute_command('AI.TENSORGET', class_key_1, 'VALUES' )
    env.assertEqual(ret[0]>=0 and ret[0]<1001, True)


def test_dagexecute_modelexecute_multidevice_resnet_ensemble_alias(env):
    if (not TEST_TF or not TEST_PT):
        return
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

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELSTORE', model_name_1, 'TF', device_1,
                              'INPUTS', 1, inputvar,
                              'OUTPUTS', 1, outputvar,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.SCRIPTSTORE', script_name_0, device_0, 'ENTRY_POINTS', 4, 'pre_process_3ch', 'pre_process_4ch', 'post_process', 'ensemble', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.SCRIPTSTORE', script_name_1, device_1, 'ENTRY_POINTS', 4, 'pre_process_3ch', 'pre_process_4ch', 'post_process', 'ensemble', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    # Cannot persist class_key_1
    check_error_message(env, con, "PERSIST key cannot be found in DAG",
                        'AI.DAGEXECUTE',
                        'PERSIST', '2', class_key_0, class_key_1, '|>',
                        'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(),
                        '|>',
                        'AI.SCRIPTEXECUTE',  script_name_0, 'pre_process_3ch',
                        'INPUTS', 1, image_key,
                        'OUTPUTS', 1, temp_key1,
                        '|>',
                        'AI.MODELEXECUTE', model_name_0,
                        'INPUTS', 1, temp_key1,
                        'OUTPUTS', 1, temp_key2_0,
                        '|>',
                        'AI.MODELEXECUTE', model_name_1,
                        'INPUTS', 1, temp_key1,
                        'OUTPUTS', 1, temp_key2_1,
                        '|>',
                        'AI.SCRIPTEXECUTE', script_name_1, 'ensemble',
                        'INPUTS', 2, temp_key2_0, temp_key2_1,
                        'OUTPUTS', 1, temp_key1,
                        '|>',
                        'AI.SCRIPTEXECUTE', script_name_0, 'post_process',
                        'INPUTS', 1, temp_key1,
                        'OUTPUTS', 1, class_key_0)

    # temp_key1 + '_foo' is an input for a DAG op which is not an output of a previous op.
    check_error_message(env, con, "INPUT key cannot be found in DAG",
                        'AI.DAGEXECUTE',
                        'PERSIST', '1', class_key_0,
                        '|>',
                        'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(),
                        '|>',
                        'AI.SCRIPTEXECUTE', script_name_0, 'pre_process_3ch',
                        'INPUTS', 1, image_key,
                        'OUTPUTS', 1, temp_key1,
                        '|>',
                        'AI.MODELEXECUTE', model_name_0,
                        'INPUTS', 1, temp_key1 + '_foo',
                        'OUTPUTS', 1, temp_key2_0,
                        '|>',
                        'AI.MODELEXECUTE', model_name_1,
                        'INPUTS', 1, temp_key1,
                        'OUTPUTS', 1, temp_key2_1,
                        '|>',
                        'AI.SCRIPTEXECUTE', script_name_1, 'ensemble',
                        'INPUTS', 2, temp_key2_0, temp_key2_1,
                        'OUTPUTS', 1, temp_key1,
                        '|>',
                        'AI.SCRIPTEXECUTE', script_name_0, 'post_process',
                        'INPUTS', 1, temp_key1,
                        'OUTPUTS', 1, class_key_0)

    # The 'ensamble' function in script_name_0 expect to receive 2 inputs (not 1)
    ret = con.execute_command(
                        'AI.DAGEXECUTE',
                        'PERSIST', '1', class_key_0, '|>',
                        'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(),
                        '|>',
                        'AI.SCRIPTEXECUTE',  script_name_0, 'pre_process_3ch',
                        'INPUTS', 1, image_key,
                        'OUTPUTS', 1, temp_key1,
                        '|>',
                        'AI.MODELEXECUTE', model_name_0,
                        'INPUTS', 1, temp_key1,
                        'OUTPUTS', 1, temp_key2_0,
                        '|>',
                        'AI.MODELEXECUTE', model_name_1,
                        'INPUTS', 1, temp_key1,
                        'OUTPUTS', 1, temp_key2_1,
                        '|>',
                        'AI.SCRIPTEXECUTE', script_name_0, 'ensemble',
                        'INPUTS', 1, temp_key2_0,
                        'OUTPUTS', 1, temp_key1,
                        '|>',
                        'AI.SCRIPTEXECUTE', script_name_0, 'post_process',
                        'INPUTS', 1, temp_key1,
                        'OUTPUTS', 1, class_key_0)

    env.assertEqual(b'OK', ret[0])
    env.assertEqual(b'OK', ret[1])
    env.assertEqual(b'OK', ret[2])
    env.assertEqual(b'OK', ret[3])
    env.assertEqual(b'NA', ret[5])
    env.assertEqual(type(ret[4]), redis.exceptions.ResponseError)
    env.assertTrue("list index out of range" in  ret[4].__str__())

    ret = con.execute_command(
        'AI.DAGEXECUTE',
                     'PERSIST', '1', class_key_0,
                     '|>',
        'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(),
                     '|>',
        'AI.SCRIPTEXECUTE',  script_name_0, 'pre_process_3ch',
                     'INPUTS', 1, image_key,
                     'OUTPUTS', 1, temp_key1,
                      '|>',
        'AI.MODELEXECUTE', model_name_0,
                     'INPUTS', 1, temp_key1,
                     'OUTPUTS', 1, temp_key2_0,
                      '|>',
        'AI.MODELEXECUTE', model_name_1,
                     'INPUTS', 1, temp_key1,
                     'OUTPUTS', 1, temp_key2_1,
                     '|>',
        'AI.SCRIPTEXECUTE', script_name_0, 'ensemble',
                      'INPUTS', 2, temp_key2_0, temp_key2_1,
                      'OUTPUTS', 1, temp_key1,
                      '|>',
        'AI.SCRIPTEXECUTE', script_name_0, 'post_process',
                      'INPUTS', 1, temp_key1,
                      'OUTPUTS', 1, class_key_0,
    )
    env.assertEqual([b'OK', b'OK', b'OK', b'OK', b'OK', b'OK'], ret)

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.TENSORGET', class_key_0, 'VALUES' )
    # tf model has 100 classes [0,999]
    env.assertEqual(ret[0]>=0 and ret[0]<1001, True)
