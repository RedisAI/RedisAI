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


def test_dag_load(env):
    con = env.getConnection()
    ret = con.execute_command(
        "AI.TENSORSET persisted_tensor_1{1} FLOAT 1 2 VALUES 5 10")
    env.assertEqual(ret, b'OK')
    command = "AI.DAGEXECUTE LOAD 1 persisted_tensor_1{1}" \
              " |> AI.TENSORGET persisted_tensor_1{1} VALUES"
    ret = con.execute_command(command)
    env.assertEqual(ret[0], [b'5', b'10'])


def test_dag_load_errors(env):
    con = env.getConnection()

    # ERR wrong number of arguments for LOAD
    check_error_message(env, con, "missing arguments after LOAD keyword in DAG command",
                        "AI.DAGEXECUTE PERSIST 2 no_tensor1{1} no_tensor2{1} LOAD 1")

    # ERR invalid or negative number of arguments for LOAD
    check_error_message(env, con, "invalid or negative value found in number of keys to LOAD",
                        "AI.DAGEXECUTE LOAD notnumber{1} |> AI.TENSORGET no_tensor{1} VALUES")

    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    # ERR number of keys to LOAD does not match the number of given arguments.
    check_error_message(env, con, "number of keys to LOAD into DAG does not match the number of given arguments",
                        "AI.DAGEXECUTE KEYS 1 a{1} LOAD 2 a{1}")

    # ERR Check key in shard succeeded  but tensor is empty.
    check_error_message(env, con, "tensor key is empty or in a different shard",
                        "AI.DAGEXECUTE KEYS 1 no_tensor{1} LOAD 1 no_tensor{1}")

    # WRONGTYPE Operation against a key holding the wrong kind of value
    con.execute_command('SET', 'no_tensor{1}', 'value')
    check_error_message(env, con, "WRONGTYPE Operation against a key holding the wrong kind of value",
                        "AI.DAGEXECUTE KEYS 1 no_tensor{1} LOAD 1 no_tensor{1}")


def test_dag_persist_errors(env):
    con = env.getConnection()
    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    # ERR wrong number of arguments for PERSIST
    check_error_message(env, con, "missing arguments after PERSIST keyword in DAG command",
                        "AI.DAGEXECUTE KEYS 1 a{1} LOAD 1 a{1} PERSIST 1")

    # ERR invalid or negative value found in number of keys to PERSIST
    check_error_message(env, con, "invalid or negative value found in number of keys to PERSIST",
                        "AI.DAGEXECUTE KEYS 1 a{1} PERSIST not_number a{1}")

    # ERR number of keys to PERSIST does not match the number of given arguments.
    check_error_message(env, con, "number of keys to PERSIST after DAG execution does not match the number of given arguments",
                        "AI.DAGEXECUTE KEYS 1 a{1} PERSIST 2 a{1}")


def test_dag_timeout_errors(env):
    con = env.getConnection()

    # ERR no value provided for timeout
    check_error_message(env, con, "No value provided for TIMEOUT",
                        "AI.DAGEXECUTE KEYS 1 a{1} PERSIST 1 a{1} TIMEOUT")

    # ERR invalid timeout value
    check_error_message(env, con, "Invalid value for TIMEOUT",
                        "AI.DAGEXECUTE KEYS 1 a{1} PERSIST 1 a{1} TIMEOUT not_number")


def test_dag_common_errors(env):
    con = env.getConnection()

    model_pb = load_file_content('graph.pb')
    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TF', DEVICE,
                          'INPUTS', 2, 'a', 'b', 'OUTPUTS', 1, 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    script = load_file_content('script.txt')
    ret = con.execute_command('AI.SCRIPTSET', 'script{1}', DEVICE, 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    # ERR bad syntax
    check_error_message(env, con, "Invalid DAG command",
                        "AI.DAGEXECUTE KEYS 1 a{1} PERSIST 1 a{1} BAD_ARG")

    # ERR DAG doesn't contains none of KEYS, LOAD, PERSIST
    check_error_message(env, con, "AI.DAGEXECUTE and AI.DAGEXECUTE_RO commands must contain at least one out of"
                                  " KEYS, LOAD, PERSIST keywords",
                        "AI.DAGEXECUTE |> AI.TENSORSET a{1} FLOAT 1 2 VALUES 5 10")

    # ERR unsupported command within DAG
    check_error_message(env, con, "Unsupported command within DAG",
                        "AI.DAGEXECUTE KEYS 1 a{1} |> AI.NOCOMMAND a{1} FLOAT 1 2 VALUES 5 10")

    # ERR wrong number of arguments for 'AI.DAGEXECUTE' command
    check_error_message(env, con, "missing arguments for 'AI.DAGEXECUTE' command", "AI.DAGEXECUTE ")

    # ERR DAG with no ops
    command = "AI.TENSORSET volatile_tensor{1} FLOAT 2 2 VALUES 5 10 5 10"
    ret = con.execute_command(command)
    env.assertEqual(ret, b'OK')
    check_error_message(env, con, "DAG is empty", "AI.DAGEXECUTE KEYS 1 volatile_tensor{1} LOAD 1 volatile_tensor{1}")

    # ERR Deprecated commands in (non-deprecated) AI.DAGEXECUTE
    check_error_message(env, con, "Deprecated AI.MODELRUN cannot be used in AI.DAGEXECUTE command",
                        "AI.DAGEXECUTE LOAD 1 volatile_tensor{1} "
                        "|> AI.MODELRUN m{1} INPUTS volatile_tensor{1} volatile_tensor{1} OUTPUTS output_tensor{1}")

    check_error_message(env, con, "Deprecated AI.SCRIPTRUN cannot be used in AI.DAGEXECUTE command",
                        "AI.DAGEXECUTE LOAD 1 volatile_tensor{1} "
                        "|> AI.SCRIPTRUN script{1} bar INPUTS volatile_tensor{1} volatile_tensor{1} OUTPUTS output_tensor{1}")


def test_dag_ro_errors(env):
    con = env.getConnection()

    # ERR wrong number of arguments for 'AI.DAGEXECUTE_RO' command
    check_error_message(env, con, "missing arguments for 'AI.DAGEXECUTE_RO' command", "AI.DAGEXECUTE_RO ")

    # ERR persist in not allowed in AI.DAGEXECUTE_RO
    check_error_message(env, con, "PERSIST cannot be specified in a read-only DAG",
                        "AI.DAGEXECUTE_RO PERSIST 1 tensor1{1} |> AI.TENSORSET tensor1{1} FLOAT 1 2 VALUES 5 10")

    # ERR AI.SCRIPTEXECUTE is not allowed in AI.DAGEXECUTE_RO
    script = load_file_content('script.txt')
    ret = con.execute_command('AI.SCRIPTSET', 'script{1}', DEVICE, 'SOURCE', script)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command("AI.TENSORSET volatile_tensor{1} FLOAT 1 2 VALUES 5 10")
    env.assertEqual(ret, b'OK')

    check_error_message(env, con, "AI.SCRIPTEXECUTE command cannot be specified in a read-only DAG",
                        "AI.DAGEXECUTE_RO LOAD 1 volatile_tensor{1}"
                        " |> AI.SCRIPTEXECUTE script{1} bar INPUTS 2 volatile_tensor{1} volatile_tensor{1}")


def test_dagrun_modelexecute_scriptexecute_resnet(env):
    if (not TEST_TF or not TEST_PT):
        return
    if(VALGRIND):
        env.debugPrint("skipping {} since it's hanging CI".format(sys._getframe().f_code.co_name), force=True)
        env.skip()
    con = env.getConnection()
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

    ret = con.execute_command('AI.SCRIPTSET', script_name, DEVICE, 'SOURCE', script)
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


def test_dag_scriptexecute_errors(env):
    if (not TEST_TF or not TEST_PT):
        return

    con = env.getConnection()
    model_name = 'imagenet_model{1}'
    script_name = 'imagenet_script{1}'
    inputvar = 'images'
    outputvar = 'output'
    model_pb, script, labels, img = load_resnet_test_data()

    ret = con.execute_command('AI.MODELSTORE', model_name, 'TF', DEVICE,
                              'INPUTS', 1, inputvar,
                              'OUTPUTS', 1, outputvar,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.SCRIPTSET', script_name, DEVICE, 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    # The function name in AI.SCRIPTEXECUTE is missing, so 'INPUTS' is considered as the function name, and
    # error is raised for the unexpected argument coming where INPUTS should come.
    image_key = 'image{1}'
    temp_key1 = 'temp_key1{1}'
    temp_key2 = 'temp_key2{1}'
    class_key = 'output{1}'
    command = (
        'AI.DAGEXECUTE', 'KEYS', 1, '{1}', '|>',
        'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(), '|>',
        'AI.SCRIPTEXECUTE',  script_name,
        'INPUTS', 1, image_key,
        'OUTPUTS', 1, temp_key1, '|>',
        'AI.MODELEXECUTE', model_name,
        'INPUTS', 1, temp_key1,
        'OUTPUTS', 1, temp_key2, '|>',
        'AI.SCRIPTEXECUTE',  script_name, 'post_process',
        'INPUTS', 1, temp_key2,
        'OUTPUTS', 1, class_key
    )
    check_error_message(env, con, "Unrecognized parameter to AI.SCRIPTEXECUTE", *command)


def test_dag_modelexecute_financialNet_errors(env):
    if not TEST_TF:
        return
    con = env.getConnection()
    model_key = 'financialNet_errors{1}'

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    ret = con.execute_command('AI.MODELSTORE', model_key, 'TF', "CPU",
                              'INPUTS', 2, 'transaction', 'reference', 'OUTPUTS', 1, 'output', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    tensor_number=1
    ret = con.execute_command('AI.TENSORSET', 'referenceTensor:{}{}'.format("{1}",tensor_number),
                                  'FLOAT', 1, 256,
                                  'BLOB', creditcard_referencedata[0].tobytes())
    env.assertEqual(ret, b'OK')

    # ERR wrong number of inputs to the model
    command = (
        'AI.DAGEXECUTE', 'LOAD', '1', 'referenceTensor:{}{}'.format("{1}", tensor_number),
                        'PERSIST', '1', 'resultTensor:{}{}'.format("{1}", tensor_number), '|>',
        'AI.TENSORSET', 'transactionTensor:{}'.format(tensor_number), 'FLOAT', 1, 30, '|>',
        'AI.MODELEXECUTE', model_key,
                        'INPUTS', 1, 'transactionTensor:{}'.format(tensor_number),
                        'OUTPUTS', 1, 'resultTensor:{}{}'.format("{1}", tensor_number), '|>',
        'AI.TENSORGET', 'resultTensor:{}{}'.format("{1}", tensor_number), 'META',
    )
    check_error_message(env, con, "Number of keys given as INPUTS here does not match model definition", *command)


def test_dag_local_tensorset(env):
    con = env.getConnection()

    command = "AI.DAGEXECUTE KEYS 1 {1} |> "\
        "AI.TENSORSET volatile_tensor1 FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORSET volatile_tensor2 FLOAT 1 2 VALUES 5 10 "

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK',b'OK'])

    # assert that transaction tensor does not exist
    ret = con.execute_command("EXISTS volatile_tensor")
    env.assertEqual(ret, 0)


def test_dagro_local_tensorset(env):
    con = env.getConnection()

    command = "AI.DAGRUN_RO |> "\
        "AI.TENSORSET volatile_tensor1 FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORSET volatile_tensor2 FLOAT 1 2 VALUES 5 10 "

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK',b'OK'])

    # assert that volatile_tensor does not exist
    ret = con.execute_command("EXISTS volatile_tensor")
    env.assertEqual(ret, 0 )


def test_dag_local_tensorset_persist(env):
    con = env.getConnection()

    command = "AI.DAGEXECUTE "\
        "PERSIST 1 tensor1{1} |> "\
        "AI.TENSORSET tensor1{1} FLOAT 1 2 VALUES 5 10"

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK'])

    # assert that PERSIST succeeded.
    ret = con.execute_command("EXISTS tensor1{1}")
    env.assertEqual(ret, 1 )

    ret = con.execute_command("AI.TENSORGET tensor1{1} META VALUES")
    env.assertEqual(ret, [b'dtype', b'FLOAT', b'shape', [1, 2], b'values', [b'5', b'10']])


def test_dag_multilocal_tensorset_persist(env):
    con = env.getConnection()

    command = "AI.DAGEXECUTE "\
        "PERSIST 1 tensor3:{1} |> "\
        "AI.TENSORSET tensor1{1} FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORSET tensor2 FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORSET tensor3:{1} FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORSET tensor4:{1} FLOAT 1 2 VALUES 5 10 "

    ret = con.execute_command(command)
    env.assertEqual([b'OK',b'OK',b'OK',b'OK'],ret)

    # assert that PERSIST succeeded.
    ret = con.execute_command("EXISTS tensor1{1}")
    env.assertEqual(ret, 0 )

    # assert that PERSIST succeeded.
    ret = con.execute_command("EXISTS tensor2")
    env.assertEqual(ret, 0 )

    # assert that PERSIST succeeded.
    ret = con.execute_command("EXISTS tensor3:{1}")
    env.assertEqual(ret, 1 )

    # assert that PERSIST succeeded.
    ret = con.execute_command("EXISTS tensor4:{1}")
    env.assertEqual(ret, 0 )

    ret = con.execute_command("AI.TENSORGET tensor3:{1} META VALUES")
    env.assertEqual(ret, [b'dtype', b'FLOAT', b'shape', [1, 2], b'values', [b'5', b'10']])


def test_dag_local_tensorset_tensorget_persist(env):
    con = env.getConnection()

    command = "AI.DAGEXECUTE PERSIST 1 tensor1{1} |> "\
        "AI.TENSORSET tensor1{1} FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORGET tensor1{1} VALUES"

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK', [b'5', b'10']])

    ret = con.execute_command("AI.TENSORGET tensor1{1} VALUES")
    env.assertEqual(ret, [b'5', b'10'])


def test_dag_local_multiple_tensorset_on_same_tensor(env):
    con = env.getConnection()

    command = "AI.DAGEXECUTE PERSIST 1 tensor1{1} |> "\
        "AI.TENSORSET tensor1{1} FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORGET tensor1{1} META VALUES |> "\
        "AI.TENSORSET tensor1{1} FLOAT 1 4 VALUES 20 40 60 80 |> "\
        "AI.TENSORGET tensor1{1} META VALUES"

    ret = con.execute_command(command)
    env.assertEqual([
                     b'OK', 
                    [b'dtype', b'FLOAT', b'shape', [1, 2], b'values', [b'5', b'10']],
                     b'OK', 
                    [b'dtype', b'FLOAT', b'shape', [1, 4], b'values', [b'20', b'40', b'60', b'80']]
                    ], ret)

    ret = con.execute_command("AI.TENSORGET tensor1{1} META VALUES")
    env.assertEqual([b'dtype', b'FLOAT', b'shape', [1, 4], b'values', [b'20', b'40',b'60',b'80']],ret)


def test_dag_load_persist_tensorset_tensorget(env):
    con = env.getConnection()

    ret = con.execute_command(
        "AI.TENSORSET persisted_tensor_1{1} FLOAT 1 2 VALUES 5 10")
    env.assertEqual(ret, b'OK')

    ret = con.execute_command(
        "AI.TENSORSET persisted_tensor_2:{1} FLOAT 1 3 VALUES 0 0 0")
    env.assertEqual(ret, b'OK')

    command = "AI.DAGEXECUTE LOAD 2 persisted_tensor_1{1} persisted_tensor_2:{1}" \
              " PERSIST 1 volatile_tensor_persisted |> "\
        "AI.TENSORSET volatile_tensor_persisted FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORGET persisted_tensor_1{1} META VALUES |> "\
        "AI.TENSORGET persisted_tensor_2:{1} META VALUES "

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK', [b'dtype', b'FLOAT', b'shape', [1, 2], b'values', [b'5', b'10']], [
                    b'dtype', b'FLOAT', b'shape', [1, 3], b'values', [b'0', b'0', b'0']]])

    ret = con.execute_command("AI.TENSORGET volatile_tensor_persisted META VALUES")
    env.assertEqual(ret, [b'dtype', b'FLOAT', b'shape', [1, 2], b'values', [b'5', b'10']])


def test_dag_keyspace_tensorget(env):
    con = env.getConnection()

    ret = con.execute_command(
        "AI.TENSORSET persisted_tensor FLOAT 1 2 VALUES 5 10")
    env.assertEqual(ret, b'OK')

    command = "AI.DAGEXECUTE LOAD 1 persisted_tensor " \
              "|> AI.TENSORGET persisted_tensor VALUES"

    ret = con.execute_command(command)
    env.assertEqual(ret, [[b'5', b'10']])


def test_dag_ro_keyspace_tensorget(env):
    con = env.getConnection()

    ret = con.execute_command(
        "AI.TENSORSET persisted_tensor FLOAT 1 2 VALUES 5 10")
    env.assertEqual(ret, b'OK')

    command = "AI.DAGEXECUTE_RO LOAD 1 persisted_tensor |> "\
        "AI.TENSORGET persisted_tensor VALUES"

    ret = con.execute_command(command)
    env.assertEqual(ret, [[b'5', b'10']])


def test_dag_keyspace_and_localcontext_tensorget(env):
    con = env.getConnection()

    ret = con.execute_command(
        "AI.TENSORSET persisted_tensor FLOAT 1 2 VALUES 5 10")
    env.assertEqual(ret, b'OK')

    command = "AI.DAGEXECUTE LOAD 1 persisted_tensor |> "\
        "AI.TENSORSET volatile_tensor FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORGET persisted_tensor VALUES |> "\
        "AI.TENSORGET volatile_tensor VALUES"

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK', [b'5', b'10'], [b'5', b'10']])


def test_dag_modelexecute_financialNet_separate_tensorget(env):
    if not TEST_TF:
        return
    con = env.getConnection()

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    model_name = 'financialNet{{hhh}}'

    ret = con.execute_command('AI.MODELSTORE', model_name, 'TF', "CPU",
                              'INPUTS', 2, 'transaction', 'reference', 'OUTPUTS', 1, 'output', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    for tensor_number in range(1,MAX_TRANSACTIONS):
        for repetition in range(1,10):
            reference_tensor = creditcard_referencedata[tensor_number]
            transaction_tensor = creditcard_transactions[tensor_number]
            result_tensor_keyname = 'resultTensor{{hhh}}{}'.format(tensor_number)
            reference_tensor_keyname = 'referenceTensor{{hhh}}{}'.format(tensor_number)
            transaction_tensor_keyname = 'transactionTensor{{hhh}}{}'.format(tensor_number)
            
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
    con = env.getConnection()

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    model_name = 'financialNet{{hhh}}'

    ret = con.execute_command('AI.MODELSTORE', model_name, 'TF', "CPU",
                              'INPUTS', 2, 'transaction', 'reference', 'OUTPUTS', 1, 'output', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    for tensor_number in range(1,MAX_TRANSACTIONS):
        for repetition in range(1,10):
            reference_tensor = creditcard_referencedata[tensor_number]
            transaction_tensor = creditcard_transactions[tensor_number]
            result_tensor_keyname = 'resultTensor{{hhh}}{}'.format(tensor_number)
            reference_tensor_keyname = 'referenceTensor{{hhh}}{}'.format(tensor_number)
            transaction_tensor_keyname = 'transactionTensor{{hhh}}{}'.format(tensor_number)
            
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
    con = env.getConnection()

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    model_name = 'financialNet{{hhh}}'

    ret = con.execute_command('AI.MODELSTORE', model_name, 'TF', 'CPU',
                              'BATCHSIZE', 2, 'MINBATCHSIZE', 2,
                              'INPUTS', 2, 'transaction', 'reference', 'OUTPUTS', 1, 'output',
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    for tensor_number in range(1,MAX_TRANSACTIONS):
        for repetition in range(1,10):
            reference_tensor = creditcard_referencedata[tensor_number]
            transaction_tensor = creditcard_transactions[tensor_number]
            result_tensor_keyname = 'resultTensor{{hhh}}{}'.format(tensor_number)
            reference_tensor_keyname = 'referenceTensor{{hhh}}{}'.format(tensor_number)
            transaction_tensor_keyname = 'transactionTensor{{hhh}}{}'.format(tensor_number)

            ret = con.execute_command('AI.TENSORSET', reference_tensor_keyname,
                                    'FLOAT', 1, 256,
                                    'BLOB', reference_tensor.tobytes())
            env.assertEqual(ret, b'OK')
            ret = con.execute_command("EXISTS {}".format(reference_tensor_keyname))
            env.assertEqual(ret, 1)

            def run():
                con = env.getConnection()
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


def test_dag_with_timeout(env):
    if not TEST_TF:
        return
    con = env.getConnection()
    batch_size = 2
    minbatch_size = 2
    timeout = 1
    model_name = 'model{1}'
    model_pb, input_var, output_var, labels, img = load_mobilenet_v2_test_data()

    con.execute_command('AI.MODELSTORE', model_name, 'TF', DEVICE,
                        'BATCHSIZE', batch_size, 'MINBATCHSIZE', minbatch_size,
                        'INPUTS', 1, input_var,
                        'OUTPUTS', 1, output_var,
                        'BLOB', model_pb)
    con.execute_command('AI.TENSORSET', 'input{1}',
                        'FLOAT', 1, img.shape[1], img.shape[0], img.shape[2],
                        'BLOB', img.tobytes())

    res = con.execute_command('AI.DAGEXECUTE',
                        'LOAD', '1', 'input{1}', 
                        'TIMEOUT', timeout, '|>',
                        'AI.MODELEXECUTE', model_name,
                        'INPUTS', 1, 'input{1}', 'OUTPUTS', 1, 'output{1}',
                        '|>', 'AI.MODELEXECUTE', model_name,
                        'INPUTS', 1, 'input{1}', 'OUTPUTS', 1, 'output{1}')

    env.assertEqual(b'TIMEDOUT', res)


def test_dag_modelexecute_financialNet_no_writes(env):
    if not TEST_TF:
        return
    con = env.getConnection()

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    model_name = 'financialNet{{hhh}}'

    ret = con.execute_command('AI.MODELSTORE', model_name, 'TF', "CPU",
                              'INPUTS', 2, 'transaction', 'reference', 'OUTPUTS', 1,'output', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    for tensor_number in range(1,MAX_TRANSACTIONS):
        for repetition in range(1,10):
            reference_tensor = creditcard_referencedata[tensor_number]
            transaction_tensor = creditcard_transactions[tensor_number]
            result_tensor_keyname = 'resultTensor{{hhh}}{}'.format(tensor_number)
            reference_tensor_keyname = 'referenceTensor{{hhh}}{}'.format(tensor_number)
            transaction_tensor_keyname = 'transactionTensor{{hhh}}{}'.format(tensor_number)
            
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
    con = env.getConnection()

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    model_name = 'financialNet_no_writes{{hhh}}'

    ret = con.execute_command('AI.MODELSTORE', model_name, 'TF', "CPU",
                              'INPUTS', 2, 'transaction', 'reference', 'OUTPUTS', 1, 'output', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    for tensor_number in range(1,MAX_TRANSACTIONS+1):
        for repetition in range(1,11):
            reference_tensor = creditcard_referencedata[tensor_number-1]
            transaction_tensor = creditcard_transactions[tensor_number-1]
            result_tensor_keyname = 'resultTensor{{hhh}}{}'.format(tensor_number)
            reference_tensor_keyname = 'referenceTensor{{hhh}}{}'.format(tensor_number)
            transaction_tensor_keyname = 'transactionTensor{{hhh}}{}'.format(tensor_number)
            
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
    con = env.getConnection()
    model_name_0 = 'imagenet_model1:{{1}}'
    model_name_1 = 'imagenet_model2:{{1}}'
    script_name = 'imagenet_script:{{1}}'
    image_key = 'image:{{1}}'
    temp_key1 = 'temp_key1:{{1}}'
    temp_key2_0 = 'temp_key2_0'
    temp_key2_1 = 'temp_key2_1'
    class_key_0 = 'output0:{{1}}'
    class_key_1 = 'output1:{{1}}'
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

    ret = con.execute_command('AI.SCRIPTSET', script_name, device_0, 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    check_error_message(env, con, "INPUT key cannot be found in DAG",
                        'AI.DAGEXECUTE', 'KEYS', '1', image_key, '|>', 'AI.SCRIPTEXECUTE',  script_name, 'pre_process_3ch',
                        'INPUTS', 1, image_key, 'OUTPUTS', 1, temp_key1)

    check_error_message(env, con, "INPUT key cannot be found in DAG",
                        'AI.DAGEXECUTE', 'KEYS', '1', image_key, '|>', 'AI.MODELEXECUTE', model_name_0,
                        'INPUTS', 1, image_key, 'OUTPUTS', 1, temp_key1)

    check_error_message(env, con, "Wrong function name given to AI.SCRIPTEXECUTE command",
                        'AI.DAGEXECUTE', 'KEYS', 1, '{{1}}',
                        '|>', 'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(),
                        '|>',
                        'AI.SCRIPTEXECUTE',  script_name, 'wrong_fn',
                        'INPUTS', 1, image_key,
                        'OUTPUTS', 1, temp_key1)

    check_error_message(env, con, "Number of keys given as INPUTS here does not match model definition",
                        'AI.DAGEXECUTE', 'KEYS', 1, '{{1}}',
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
    con = env.getConnection()

    model_name_0 = 'imagenet_model1:{{1}}'
    model_name_1 = 'imagenet_model2:{{1}}'
    script_name_0 = 'imagenet_script1:{{1}}'
    script_name_1 = 'imagenet_script2:{{1}}'
    inputvar = 'images'
    outputvar = 'output'
    image_key = 'image:{{1}}'
    temp_key1 = 'temp_key1:{{1}}'
    temp_key2_0 = 'temp_key2_0'
    temp_key2_1 = 'temp_key2_1'
    class_key_0 = 'output0:{{1}}'
    class_key_1 = 'output1:{{1}}'

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

    ret = con.execute_command('AI.SCRIPTSET', script_name_0, device_0, 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.SCRIPTSET', script_name_1, device_1, 'SOURCE', script)
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
    check_error_message(env, con, "Wrong number of inputs provided to AI.SCRIPTEXECUTE command",
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
