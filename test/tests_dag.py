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
        "AI.TENSORSET persisted_tensor_1{{1}} FLOAT 1 2 VALUES 5 10")
    env.assertEqual(ret, b'OK')
    command = "AI.DAGRUN "\
                "LOAD 1 persisted_tensor_1{{1}} "\
                "PERSIST 1 tensor1{{1}} |> "\
              "AI.TENSORSET tensor1{{1}} FLOAT 1 2 VALUES 5 10"

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK'])

def test_dag_load_errors(env):
    con = env.getConnection()

    # ERR tensor key is empty
    try:
        command = "AI.DAGRUN "\
                    "LOAD 1 persisted_tensor_1{{1}} "\
                    "PERSIST 1 tensor1{{1}} |> "\
                "AI.TENSORSET tensor1{{1}} FLOAT 1 2 VALUES 5 10"

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("tensor key is empty",exception.__str__())

    # WRONGTYPE Operation against a key holding the wrong kind of value
    try:
        con.execute_command('SET', 'non-tensor{{1}}', 'value')
        command = "AI.DAGRUN "\
                    "LOAD 1 non-tensor{{1}} "\
                    "PERSIST 1 tensor1{{1}} |> "\
                "AI.TENSORSET tensor1{{1}} FLOAT 1 2 VALUES 5 10"

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("WRONGTYPE Operation against a key holding the wrong kind of value",exception.__str__())


def test_dag_common_errors(env):
    con = env.getConnection()

    # ERR unsupported command within DAG
    try:
        command = "AI.DAGRUN |> "\
                "AI.DONTEXIST tensor1{{1}} FLOAT 1 2 VALUES 5 10"

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("ERR unsupported command within DAG",exception.__str__())

    # ERR wrong number of arguments for 'AI.DAGRUN' command
    try:
        command = "AI.DAGRUN "

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("wrong number of arguments for 'AI.DAGRUN' command",exception.__str__())

    # ERR invalid or negative value found in number of keys to PERSIST
    try:
        command = "AI.DAGRUN PERSIST notnumber{{1}} |> "\
                  "AI.TENSORSET tensor1 FLOAT 1 2 VALUES 5 10"

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("invalid or negative value found in number of keys to PERSIST",exception.__str__())

    # ERR invalid or negative value found in number of keys to LOAD
    try:
        command = "AI.DAGRUN LOAD notnumber{{1}} |> "\
                "AI.TENSORSET tensor1 FLOAT 1 2 VALUES 5 10"

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("invalid or negative value found in number of keys to LOAD",exception.__str__())


def test_dagro_common_errors(env):
    con = env.getConnection()

    # ERR unsupported command within DAG
    try:
        command = "AI.DAGRUN_RO |> "\
                "AI.DONTEXIST tensor1 FLOAT 1 2 VALUES 5 10"

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("ERR unsupported command within DAG",exception.__str__())

    # ERR wrong number of arguments for 'AI.DAGRUN' command
    try:
        command = "AI.DAGRUN_RO "

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("wrong number of arguments for 'AI.DAGRUN_RO' command",exception.__str__())

    # ERR invalid or negative value found in number of keys to LOAD
    try:
        command = "AI.DAGRUN_RO LOAD notnumber{{1}} |> "\
                "AI.TENSORSET tensor1{{1}} FLOAT 1 2 VALUES 5 10"

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("invalid or negative value found in number of keys to LOAD",exception.__str__())


def test_dag_modelrun_financialNet_errors(env):
    if not TEST_TF:
        return
    con = env.getConnection()
    model_key = 'financialNet_errors{{1}}'

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    ret = con.execute_command('AI.MODELSET', model_key, 'TF', "CPU",
                              'INPUTS', 'transaction', 'reference', 'OUTPUTS', 'output', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    tensor_number=1
    ret = con.execute_command(  'AI.TENSORSET', 'referenceTensor:{{1}}{0}'.format(tensor_number),
                                  'FLOAT', 1, 256,
                                  'BLOB', creditcard_referencedata[0].tobytes())
    env.assertEqual(ret, b'OK')

    # ERR wrong number of inputs
    try:
        ret = con.execute_command(
        'AI.DAGRUN', 'LOAD', '1', 'referenceTensor:{{1}}{}'.format(tensor_number), 
                        'PERSIST', '1', 'resultTensor:{{1}}{}'.format(tensor_number), '|>',
        'AI.TENSORSET', 'transactionTensor:{}'.format(tensor_number), 'FLOAT', 1, 30, '|>',
        'AI.MODELRUN', model_key, 
                        'INPUTS', 'transactionTensor:{}'.format(tensor_number),
                        'OUTPUTS', 'resultTensor:{{1}}{}'.format(tensor_number), '|>',
        'AI.TENSORGET', 'resultTensor:{{1}}{}'.format(tensor_number), 'META',
    )
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("ERR unsupported command within DAG",exception.__str__())


def test_dag_local_tensorset(env):
    con = env.getConnection()

    command = "AI.DAGRUN "\
        "AI.TENSORSET volatile_tensor1 FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORSET volatile_tensor2 FLOAT 1 2 VALUES 5 10 "

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK',b'OK'])

    # assert that transaction tensor does not exist
    ret = con.execute_command("EXISTS volatile_tensor")
    env.assertEqual(ret, 0 )


def test_dagro_local_tensorset(env):
    con = env.getConnection()

    command = "AI.DAGRUN_RO "\
        "AI.TENSORSET volatile_tensor1 FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORSET volatile_tensor2 FLOAT 1 2 VALUES 5 10 "

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK',b'OK'])

    # assert that transaction tensor does not exist
    ret = con.execute_command("EXISTS volatile_tensor")
    env.assertEqual(ret, 0 )


def test_dag_local_tensorset_persist(env):
    con = env.getConnection()

    command = "AI.DAGRUN "\
        "PERSIST 1 tensor1{{1}} |> "\
        "AI.TENSORSET tensor1{{1}} FLOAT 1 2 VALUES 5 10"

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK'])

    # assert that transaction tensor exists
    ret = con.execute_command("EXISTS tensor1{{1}}")
    env.assertEqual(ret, 1 )

    ret = con.execute_command("AI.TENSORGET tensor1{{1}} META VALUES")
    env.assertEqual(ret, [b'dtype', b'FLOAT', b'shape', [1, 2], b'values', [b'5', b'10']])


def test_dagro_local_tensorset_persist(env):
    con = env.getConnection()

    command = "AI.DAGRUN_RO "\
        "PERSIST 1 tensor1{{1}} |> "\
        "AI.TENSORSET tensor1{{1}} FLOAT 1 2 VALUES 5 10"

    try:
        con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("PERSIST cannot be specified in a read-only DAG", exception.__str__())


def test_dag_multilocal_tensorset_persist(env):
    con = env.getConnection()

    command = "AI.DAGRUN "\
        "PERSIST 1 tensor3:{{1}} |> "\
        "AI.TENSORSET tensor1{{1}} FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORSET tensor2 FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORSET tensor3:{{1}} FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORSET tensor4:{{1}} FLOAT 1 2 VALUES 5 10 "

    ret = con.execute_command(command)
    env.assertEqual([b'OK',b'OK',b'OK',b'OK'],ret)

    # assert that transaction tensor exists
    ret = con.execute_command("EXISTS tensor1{{1}}")
    env.assertEqual(ret, 0 )

    # assert that transaction tensor exists
    ret = con.execute_command("EXISTS tensor2")
    env.assertEqual(ret, 0 )

    # assert that transaction tensor exists
    ret = con.execute_command("EXISTS tensor3:{{1}}")
    env.assertEqual(ret, 1 )

    # assert that transaction tensor exists
    ret = con.execute_command("EXISTS tensor4:{{1}}")
    env.assertEqual(ret, 0 )

    ret = con.execute_command("AI.TENSORGET tensor3:{{1}} META VALUES")
    env.assertEqual(ret, [b'dtype', b'FLOAT', b'shape', [1, 2], b'values', [b'5', b'10']])


def test_dag_local_tensorset_tensorget_persist(env):
    con = env.getConnection()

    command = "AI.DAGRUN PERSIST 1 tensor1{{1}} |> "\
        "AI.TENSORSET tensor1{{1}} FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORGET tensor1{{1}} VALUES"

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK', [b'5', b'10']])

    ret = con.execute_command("AI.TENSORGET tensor1{{1}} VALUES")
    env.assertEqual(ret, [b'5', b'10'])


def test_dag_local_multiple_tensorset_on_same_tensor(env):
    con = env.getConnection()

    command = "AI.DAGRUN "\
                     "PERSIST 1 tensor1{{1}} |> "\
        "AI.TENSORSET tensor1{{1}} FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORGET tensor1{{1}} META VALUES |> "\
        "AI.TENSORSET tensor1{{1}} FLOAT 1 4 VALUES 20 40 60 80 |> "\
        "AI.TENSORGET tensor1{{1}} META VALUES"

    ret = con.execute_command(command)
    env.assertEqual([
                     b'OK', 
                    [b'dtype', b'FLOAT', b'shape', [1, 2], b'values', [b'5', b'10']],
                     b'OK', 
                    [b'dtype', b'FLOAT', b'shape', [1, 4], b'values', [b'20', b'40', b'60', b'80']]
                    ], ret)

    ret = con.execute_command("AI.TENSORGET tensor1{{1}} META VALUES")
    env.assertEqual([b'dtype', b'FLOAT', b'shape', [1, 4], b'values', [b'20', b'40',b'60',b'80']],ret)


def test_dag_load_persist_tensorset_tensorget(env):
    con = env.getConnection()

    ret = con.execute_command(
        "AI.TENSORSET persisted_tensor_1{{1}} FLOAT 1 2 VALUES 5 10")
    env.assertEqual(ret, b'OK')

    ret = con.execute_command(
        "AI.TENSORSET persisted_tensor_2:{{1}} FLOAT 1 3 VALUES 0 0 0")
    env.assertEqual(ret, b'OK')

    command = "AI.DAGRUN LOAD 2 persisted_tensor_1{{1}} persisted_tensor_2:{{1}} PERSIST 1 volatile_tensor_persisted |> "\
        "AI.TENSORSET volatile_tensor_persisted FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORGET persisted_tensor_1{{1}} META VALUES |> "\
        "AI.TENSORGET persisted_tensor_2:{{1}} META VALUES "

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK', [b'dtype', b'FLOAT', b'shape', [1, 2], b'values', [b'5', b'10']], [
                    b'dtype', b'FLOAT', b'shape', [1, 3], b'values', [b'0', b'0', b'0']]])

    ret = con.execute_command("AI.TENSORGET volatile_tensor_persisted META VALUES")
    env.assertEqual(ret, [b'dtype', b'FLOAT', b'shape', [1, 2], b'values', [b'5', b'10']])


def test_dag_local_tensorset_tensorget(env):
    con = env.getConnection()

    command = "AI.DAGRUN "\
        "AI.TENSORSET volatile_tensor FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORGET volatile_tensor META VALUES"

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK', [b'dtype', b'FLOAT', b'shape', [1, 2], b'values', [b'5', b'10']]])


def test_dag_keyspace_tensorget(env):
    con = env.getConnection()

    ret = con.execute_command(
        "AI.TENSORSET persisted_tensor FLOAT 1 2 VALUES 5 10")
    env.assertEqual(ret, b'OK')

    command = "AI.DAGRUN LOAD 1 persisted_tensor |> "\
        "AI.TENSORGET persisted_tensor VALUES"

    ret = con.execute_command(command)
    env.assertEqual(ret, [[b'5', b'10']])


def test_dagro_keyspace_tensorget(env):
    con = env.getConnection()

    ret = con.execute_command(
        "AI.TENSORSET persisted_tensor FLOAT 1 2 VALUES 5 10")
    env.assertEqual(ret, b'OK')

    command = "AI.DAGRUN_RO LOAD 1 persisted_tensor |> "\
        "AI.TENSORGET persisted_tensor VALUES"

    ret = con.execute_command(command)
    env.assertEqual(ret, [[b'5', b'10']])


def test_dag_keyspace_and_localcontext_tensorget(env):
    con = env.getConnection()

    ret = con.execute_command(
        "AI.TENSORSET persisted_tensor FLOAT 1 2 VALUES 5 10")
    env.assertEqual(ret, b'OK')

    command = "AI.DAGRUN LOAD 1 persisted_tensor |> "\
        "AI.TENSORSET volatile_tensor FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORGET persisted_tensor VALUES |> "\
        "AI.TENSORGET volatile_tensor VALUES"

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK', [b'5', b'10'], [b'5', b'10']])


def test_dag_modelrun_financialNet_separate_tensorget(env):
    if not TEST_TF:
        return
    con = env.getConnection()

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    model_name = 'financialNet{{hhh}}'

    ret = con.execute_command('AI.MODELSET', model_name, 'TF', "CPU",
                              'INPUTS', 'transaction', 'reference', 'OUTPUTS', 'output', 'BLOB', model_pb)
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
                'AI.DAGRUN', 'LOAD', '1', reference_tensor_keyname, 
                'PERSIST', '1', result_tensor_keyname, '|>',
                'AI.TENSORSET', transaction_tensor_keyname, 'FLOAT', 1, 30,'BLOB', transaction_tensor.tobytes(), '|>',
                'AI.MODELRUN', model_name, 
                    'INPUTS', transaction_tensor_keyname, reference_tensor_keyname,
                    'OUTPUTS', result_tensor_keyname, 
            )
            env.assertEqual([b'OK',b'OK'],ret)

            ret = con.execute_command("AI.TENSORGET {} META".format(
                result_tensor_keyname))
            env.assertEqual([b'dtype', b'FLOAT', b'shape', [1, 2]], ret)


def test_dag_modelrun_financialNet(env):
    if not TEST_TF:
        return
    con = env.getConnection()

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    model_name = 'financialNet{{hhh}}'

    ret = con.execute_command('AI.MODELSET', model_name, 'TF', "CPU",
                              'INPUTS', 'transaction', 'reference', 'OUTPUTS', 'output', 'BLOB', model_pb)
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
                'AI.DAGRUN', 'LOAD', '1', reference_tensor_keyname, 
                            'PERSIST', '1', result_tensor_keyname, '|>',
                'AI.TENSORSET', transaction_tensor_keyname, 'FLOAT', 1, 30,'BLOB', transaction_tensor.tobytes(), '|>',
                'AI.MODELRUN', model_name, 
                            'INPUTS', transaction_tensor_keyname, reference_tensor_keyname,
                            'OUTPUTS', result_tensor_keyname, '|>',
                'AI.TENSORGET', result_tensor_keyname, 'META',
            )
            env.assertEqual([b'OK',b'OK',[b'dtype', b'FLOAT', b'shape', [1, 2]]], ret)

            # assert that transaction tensor does not exist
            ret = con.execute_command("EXISTS {}".format(transaction_tensor_keyname))
            env.assertEqual(ret, 0)
            # assert that result tensor exists
            ret = con.execute_command("EXISTS {}".format(result_tensor_keyname))
            env.assertEqual(ret, 1)

def test_dag_modelrun_financialNet_no_writes(env):
    if not TEST_TF:
        return
    con = env.getConnection()

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    model_name = 'financialNet{{hhh}}'

    ret = con.execute_command('AI.MODELSET', model_name, 'TF', "CPU",
                              'INPUTS', 'transaction', 'reference', 'OUTPUTS', 'output', 'BLOB', model_pb)
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
                'AI.DAGRUN', 'LOAD', '1', reference_tensor_keyname, '|>',
                'AI.TENSORSET', transaction_tensor_keyname, 'FLOAT', 1, 30,'BLOB', transaction_tensor.tobytes(), '|>',
                'AI.MODELRUN', model_name, 
                            'INPUTS', transaction_tensor_keyname, reference_tensor_keyname,
                            'OUTPUTS', result_tensor_keyname, '|>',
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


def test_dagro_modelrun_financialNet_no_writes_multiple_modelruns(env):
    if not TEST_TF:
        return
    con = env.getConnection()

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    model_name = 'financialNet_no_writes{{hhh}}'

    ret = con.execute_command('AI.MODELSET', model_name, 'TF', "CPU",
                              'INPUTS', 'transaction', 'reference', 'OUTPUTS', 'output', 'BLOB', model_pb)
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
                'AI.DAGRUN_RO', 'LOAD', '1', reference_tensor_keyname, '|>',
                'AI.TENSORSET', transaction_tensor_keyname, 'FLOAT', 1, 30,'BLOB', transaction_tensor.tobytes(), '|>',
                'AI.MODELRUN', model_name, 
                            'INPUTS', transaction_tensor_keyname, reference_tensor_keyname,
                            'OUTPUTS', result_tensor_keyname, '|>',
                'AI.TENSORGET', result_tensor_keyname, 'META', 'VALUES', '|>',
                'AI.MODELRUN', model_name, 
                            'INPUTS', transaction_tensor_keyname, reference_tensor_keyname,
                            'OUTPUTS', result_tensor_keyname, '|>',
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
    env.assertEqual(0, financialNetRunInfo['samples'])
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
