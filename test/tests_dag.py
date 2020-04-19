import redis

from includes import *

'''
python -m RLTest --test tests_dag.py --module path/to/redisai.so
'''


def test_dag_load(env):
    con = env.getConnection()
    ret = con.execute_command(
        "AI.TENSORSET persisted_tensor_1 FLOAT 1 2 VALUES 5 10")
    env.assertEqual(ret, b'OK')
    command = "AI.DAGRUN "\
                "LOAD 1 persisted_tensor_1 "\
                "PERSIST 1 tensor1 |> "\
              "AI.TENSORSET tensor1 FLOAT 1 2 VALUES 5 10"

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK'])

def test_dag_load_errors(env):
    con = env.getConnection()

    # ERR tensor key is empty
    try:
        command = "AI.DAGRUN "\
                    "LOAD 1 persisted_tensor_1 "\
                    "PERSIST 1 tensor1 |> "\
                "AI.TENSORSET tensor1 FLOAT 1 2 VALUES 5 10"

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("tensor key is empty",exception.__str__())

    # WRONGTYPE Operation against a key holding the wrong kind of value
    try:
        con.execute_command('SET', 'non-tensor', 'value')
        command = "AI.DAGRUN "\
                    "LOAD 1 non-tensor "\
                    "PERSIST 1 tensor1 |> "\
                "AI.TENSORSET tensor1 FLOAT 1 2 VALUES 5 10"

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
                "AI.DONTEXIST tensor1 FLOAT 1 2 VALUES 5 10"

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
        command = "AI.DAGRUN PERSIST notnumber |> "\
                "AI.TENSORSET tensor1 FLOAT 1 2 VALUES 5 10"

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("invalid or negative value found in number of keys to PERSIST",exception.__str__())

    # ERR invalid or negative value found in number of keys to LOAD
    try:
        command = "AI.DAGRUN LOAD notnumber |> "\
                "AI.TENSORSET tensor1 FLOAT 1 2 VALUES 5 10"

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("invalid or negative value found in number of keys to LOAD",exception.__str__())


def test_dag_modelrun_financialNet_errors(env):
    con = env.getConnection()

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    ret = con.execute_command('AI.MODELSET', 'financialNet', 'TF', "CPU",
                              'INPUTS', 'transaction', 'reference', 'OUTPUTS', 'output', model_pb)
    env.assertEqual(ret, b'OK')

    tensor_number=1
    ret = con.execute_command(  'AI.TENSORSET', 'referenceTensor:{0}'.format(tensor_number),
                                  'FLOAT', 1, 256,
                                  'BLOB', creditcard_referencedata[0].tobytes())
    env.assertEqual(ret, b'OK')
    

    # ERR wrong number of inputs
    try:
        tensor_number=1
        ret = con.execute_command(
        'AI.DAGRUN', 'LOAD', '1', 'referenceTensor:{}'.format(tensor_number), 
                        'PERSIST', '1', 'classificationTensor:{}'.format(tensor_number), '|>',
        'AI.TENSORSET', 'transactionTensor:{}'.format(tensor_number), 'FLOAT', 1, 30, '|>',
        'AI.MODELRUN', 'financialNet', 
                        'INPUTS', 'transactionTensor:{}'.format(tensor_number),
                        'OUTPUTS', 'classificationTensor:{}'.format(tensor_number), '|>',
        'AI.TENSORGET', 'classificationTensor:{}'.format(tensor_number), 'META',
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

def test_dag_local_tensorset_persist(env):
    con = env.getConnection()

    command = "AI.DAGRUN "\
        "PERSIST 1 tensor1 |> "\
        "AI.TENSORSET tensor1 FLOAT 1 2 VALUES 5 10"

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK'])

    # assert that transaction tensor exists
    ret = con.execute_command("EXISTS tensor1")
    env.assertEqual(ret, 1 )

    ret = con.execute_command("AI.TENSORGET tensor1 META VALUES")
    env.assertEqual(ret, [b'dtype', b'FLOAT', b'shape', [1, 2], b'values', [b'5', b'10']])


def test_dag_multilocal_tensorset_persist(env):
    con = env.getConnection()

    command = "AI.DAGRUN "\
        "PERSIST 1 tensor3 |> "\
        "AI.TENSORSET tensor1 FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORSET tensor2 FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORSET tensor3 FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORSET tensor4 FLOAT 1 2 VALUES 5 10 "

    ret = con.execute_command(command)
    env.assertEqual([b'OK',b'OK',b'OK',b'OK'],ret)

    # assert that transaction tensor exists
    ret = con.execute_command("EXISTS tensor1")
    env.assertEqual(ret, 0 )

    # assert that transaction tensor exists
    ret = con.execute_command("EXISTS tensor2")
    env.assertEqual(ret, 0 )

    # assert that transaction tensor exists
    ret = con.execute_command("EXISTS tensor3")
    env.assertEqual(ret, 1 )

    # assert that transaction tensor exists
    ret = con.execute_command("EXISTS tensor4")
    env.assertEqual(ret, 0 )

    ret = con.execute_command("AI.TENSORGET tensor3 META VALUES")
    env.assertEqual(ret, [b'dtype', b'FLOAT', b'shape', [1, 2], b'values', [b'5', b'10']])

def test_dag_local_tensorset_tensorget_persist(env):
    con = env.getConnection()

    command = "AI.DAGRUN PERSIST 1 tensor1 |> "\
        "AI.TENSORSET tensor1 FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORGET tensor1 VALUES"

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK', [b'5', b'10']])

    ret = con.execute_command("AI.TENSORGET tensor1 VALUES")
    env.assertEqual(ret, [b'5', b'10'])


def test_dag_local_multiple_tensorset_on_same_tensor(env):
    con = env.getConnection()

    command = "AI.DAGRUN "\
                     "PERSIST 1 tensor1 |> "\
        "AI.TENSORSET tensor1 FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORGET tensor1 META VALUES |> "\
        "AI.TENSORSET tensor1 FLOAT 1 4 VALUES 20 40 60 80 |> "\
        "AI.TENSORGET tensor1 META VALUES"

    ret = con.execute_command(command)
    env.assertEqual([
                     b'OK', 
                    [b'dtype', b'FLOAT', b'shape', [1, 2], b'values', [b'5', b'10']],
                     b'OK', 
                    [b'dtype', b'FLOAT', b'shape', [1, 4], b'values', [b'20', b'40', b'60', b'80']]
                    ], ret)

    ret = con.execute_command("AI.TENSORGET tensor1 META VALUES")
    env.assertEqual([b'dtype', b'FLOAT', b'shape', [1, 4], b'values', [b'20', b'40',b'60',b'80']],ret)


def test_dag_load_persist_tensorset_tensorget(env):
    con = env.getConnection()

    ret = con.execute_command(
        "AI.TENSORSET persisted_tensor_1 FLOAT 1 2 VALUES 5 10")
    env.assertEqual(ret, b'OK')

    ret = con.execute_command(
        "AI.TENSORSET persisted_tensor_2 FLOAT 1 3 VALUES 0 0 0")
    env.assertEqual(ret, b'OK')

    command = "AI.DAGRUN LOAD 2 persisted_tensor_1 persisted_tensor_2 PERSIST 1 volatile_tensor_persisted |> "\
        "AI.TENSORSET volatile_tensor_persisted FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORGET persisted_tensor_1 META VALUES |> "\
        "AI.TENSORGET persisted_tensor_2 META VALUES "

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
    con = env.getConnection()

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    ret = con.execute_command('AI.MODELSET', 'financialNet', 'TF', "CPU",
                              'INPUTS', 'transaction', 'reference', 'OUTPUTS', 'output', model_pb)
    env.assertEqual(ret, b'OK')

    tensor_number = 1
    for reference_tensor in creditcard_referencedata[:5]:
        ret = con.execute_command('AI.TENSORSET', 'referenceTensor:{0}'.format(tensor_number),
                                  'FLOAT', 1, 256,
                                  'BLOB', reference_tensor.tobytes())
        env.assertEqual(ret, b'OK')
        tensor_number = tensor_number + 1

    tensor_number = 1
    for transaction_tensor in creditcard_transactions[:5]:
        ret = con.execute_command(
            'AI.DAGRUN', 'LOAD', '1', 'referenceTensor:{}'.format(tensor_number), 
            'PERSIST', '1', 'classificationTensor:{}'.format(tensor_number), '|>',
            'AI.TENSORSET', 'transactionTensor:{}'.format(tensor_number), 'FLOAT', 1, 30,'BLOB', transaction_tensor.tobytes(), '|>',
            'AI.MODELRUN', 'financialNet', 
                'INPUTS', 'transactionTensor:{}'.format(tensor_number), 'referenceTensor:{}'.format(tensor_number),
                'OUTPUTS', 'classificationTensor:{}'.format(tensor_number), 
        )
        env.assertEqual([b'OK',b'OK'],ret)

        ret = con.execute_command("AI.TENSORGET classificationTensor:{} META".format(
            tensor_number))
        env.assertEqual([b'dtype', b'FLOAT', b'shape', [1, 2]], ret)

        # assert that transaction tensor does not exist
        ret = con.execute_command("EXISTS transactionTensor:{} META".format(
            tensor_number))
        env.assertEqual(ret, 0 )
        tensor_number = tensor_number + 1

def test_dag_modelrun_financialNet(env):
    con = env.getConnection()

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    ret = con.execute_command('AI.MODELSET', 'financialNet', 'TF', "CPU",
                              'INPUTS', 'transaction', 'reference', 'OUTPUTS', 'output', model_pb)
    env.assertEqual(ret, b'OK')

    tensor_number = 1
    for reference_tensor in creditcard_referencedata[:5]:
        ret = con.execute_command('AI.TENSORSET', 'referenceTensor:{0}'.format(tensor_number),
                                  'FLOAT', 1, 256,
                                  'BLOB', reference_tensor.tobytes())
        env.assertEqual(ret, b'OK')
        tensor_number = tensor_number + 1

    tensor_number = 1
    for transaction_tensor in creditcard_transactions[:5]:
        ret = con.execute_command(
            'AI.DAGRUN', 'LOAD', '1', 'referenceTensor:{}'.format(tensor_number), 
                         'PERSIST', '1', 'classificationTensor:{}'.format(tensor_number), '|>',
            'AI.TENSORSET', 'transactionTensor:{}'.format(tensor_number), 'FLOAT', 1, 30,'BLOB', transaction_tensor.tobytes(), '|>',
            'AI.MODELRUN', 'financialNet', 
                           'INPUTS', 'transactionTensor:{}'.format(tensor_number), 'referenceTensor:{}'.format(tensor_number),
                           'OUTPUTS', 'classificationTensor:{}'.format(tensor_number), '|>',
            'AI.TENSORGET', 'classificationTensor:{}'.format(tensor_number), 'META',
        )
        env.assertEqual([b'OK',b'OK',[b'dtype', b'FLOAT', b'shape', [1, 2]]], ret)

        # assert that transaction tensor does not exist
        ret = con.execute_command("EXISTS transactionTensor:{}".format(
            tensor_number))
        env.assertEqual(ret, 0 )
        tensor_number = tensor_number + 1

def test_dag_modelrun_financialNet_no_writes(env):
    con = env.getConnection()

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    ret = con.execute_command('AI.MODELSET', 'financialNet', 'TF', "CPU",
                              'INPUTS', 'transaction', 'reference', 'OUTPUTS', 'output', model_pb)
    env.assertEqual(ret, b'OK')

    tensor_number = 1
    for reference_tensor in creditcard_referencedata:
        ret = con.execute_command('AI.TENSORSET', 'referenceTensor:{0}'.format(tensor_number),
                                  'FLOAT', 1, 256,
                                  'BLOB', reference_tensor.tobytes())
        env.assertEqual(ret, b'OK')
        tensor_number = tensor_number + 1

    tensor_number = 1
    for transaction_tensor in creditcard_transactions:
        for run_number in range(1,10):
            ret = con.execute_command(
                'AI.DAGRUN', 'LOAD', '1', 'referenceTensor:{}'.format(tensor_number), '|>',
                'AI.TENSORSET', 'transactionTensor:{}'.format(tensor_number), 'FLOAT', 1, 30,'BLOB', transaction_tensor.tobytes(), '|>',
                'AI.MODELRUN', 'financialNet', 
                            'INPUTS', 'transactionTensor:{}'.format(tensor_number), 'referenceTensor:{}'.format(tensor_number),
                            'OUTPUTS', 'classificationTensor:{}'.format(tensor_number), '|>',
                'AI.TENSORGET', 'classificationTensor:{}'.format(tensor_number), 'META',  '|>',
                'AI.TENSORGET', 'classificationTensor:{}'.format(tensor_number), 'VALUES'
            )
            env.assertEqual(4, len(ret))
            env.assertEqual([b'OK', b'OK'], ret[:2])
            env.assertEqual([b'dtype', b'FLOAT', b'shape', [1, 2]], ret[2])
            values = ret[3]
            # Assert that resulting classification is within [0,1]
            env.assertEqual(True, 0 <= float(values[0]) <= 1)
            env.assertEqual(True, 0 <= float(values[1]) <= 1)

            # assert that transactionTensor does not exist
            ret = con.execute_command("EXISTS transactionTensor:{}".format(
                tensor_number))
            env.assertEqual(ret, 0 )

            # assert that classificationTensor does not exist
            ret = con.execute_command("EXISTS classificationTensor:{}".format(
                tensor_number))
            env.assertEqual(ret, 0 )
        tensor_number = tensor_number + 1


def test_dag_modelrun_financialNet_no_writes_multiple_modelruns(env):
    con = env.getConnection()

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    ret = con.execute_command('AI.MODELSET', 'financialNet', 'TF', "CPU",
                              'INPUTS', 'transaction', 'reference', 'OUTPUTS', 'output', model_pb)
    env.assertEqual(ret, b'OK')

    tensor_number = 1
    for reference_tensor in creditcard_referencedata:
        ret = con.execute_command('AI.TENSORSET', 'referenceTensor:{0}'.format(tensor_number),
                                  'FLOAT', 1, 256,
                                  'BLOB', reference_tensor.tobytes())
        env.assertEqual(ret, b'OK')
        tensor_number = tensor_number + 1

    tensor_number = 1
    for transaction_tensor in creditcard_transactions:
        ret = con.execute_command(
            'AI.DAGRUN', 'LOAD', '1', 'referenceTensor:{}'.format(tensor_number), '|>',
            'AI.TENSORSET', 'transactionTensor:{}'.format(tensor_number), 'FLOAT', 1, 30,'BLOB', transaction_tensor.tobytes(), '|>',
            'AI.MODELRUN', 'financialNet', 
                           'INPUTS', 'transactionTensor:{}'.format(tensor_number), 'referenceTensor:{}'.format(tensor_number),
                           'OUTPUTS', 'classificationTensor:{}'.format(tensor_number), '|>',
            'AI.TENSORGET', 'classificationTensor:{}'.format(tensor_number), 'META', 'VALUES', '|>',
            'AI.MODELRUN', 'financialNet', 
                           'INPUTS', 'transactionTensor:{}'.format(tensor_number), 'referenceTensor:{}'.format(tensor_number),
                           'OUTPUTS', 'classificationTensor:{}'.format(tensor_number), '|>',
            'AI.TENSORGET', 'classificationTensor:{}'.format(tensor_number), 'META', 'VALUES', 
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

        # assert that transactionTensor does not exist
        ret = con.execute_command("EXISTS transactionTensor:{}".format(
            tensor_number))
        env.assertEqual(ret, 0)

        # assert that classificationTensor does not exist
        ret = con.execute_command("EXISTS classificationTensor:{}".format(
            tensor_number))
        env.assertEqual(ret, 0)
        tensor_number = tensor_number + 1
