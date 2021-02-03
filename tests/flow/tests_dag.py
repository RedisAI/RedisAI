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

    # ERR wrong number of arguments for LOAD
    try:
        command = "AI.DAGRUN PERSIST 1 no_tensor{1} LOAD 1"

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("wrong number of arguments for LOAD in 'AI.DAGRUN' command",exception.__str__())

    # ERR invalid or negative number of arguments for LOAD
    try:
        command = "AI.DAGRUN LOAD notnumber{{1}} |> AI.TENSORGET 'no_tensor{1}'"

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("invalid or negative value found in number of keys to LOAD",exception.__str__())

    con.execute_command('AI.TENSORSET', 'a{{1}}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    # ERR number of keys to LOAD does not match the number of given arguments.
    try:
        command = "AI.DAGRUN LOAD 2 a{{1}}"

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("number of keys to LOAD in AI.DAGRUN command does not match the number of "
                        "given arguments", exception.__str__())

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


def test_dag_persist_errors(env):
    con = env.getConnection()
    con.execute_command('AI.TENSORSET', 'a{{1}}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    # ERR wrong number of arguments for PERSIST
    try:
        command = "AI.DAGRUN LOAD 1 a{{1}} PERSIST 1"

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("wrong number of arguments for PERSIST in 'AI.DAGRUN' command",exception.__str__())

    # ERR invalid or negative value found in number of keys to PERSIST
    try:
        command = "AI.DAGRUN PERSIST notnumber{{1}} |> " \
                  "AI.TENSORSET tensor1 FLOAT 1 2 VALUES 5 10"

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("invalid or negative value found in number of keys to PERSIST",exception.__str__())

    # ERR number of keys to PERSIST does not match the number of given arguments.
    try:
        command = "AI.DAGRUN PERSIST 2 tensor1{1}"

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("number of keys to PERSIST in AI.DAGRUN command does not match the number of "
                        "given arguments", exception.__str__())


def test_dag_timeout_errors(env):
    con = env.getConnection()

    # ERR no value provided for timeout
    try:
        command = "AI.DAGRUN PERSIST 1 no_tensor{1} TIMEOUT"

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("No value provided for TIMEOUT",exception.__str__())

    # ERR invalid timeout value
    try:
        command = "AI.DAGRUN TIMEOUT notnumber{{1}} |> " \
                  "AI.TENSORSET tensor1 FLOAT 1 2 VALUES 5 10"

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Invalid value for TIMEOUT",exception.__str__())


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
        env.assertEqual("unsupported command within DAG",exception.__str__())

    # ERR wrong number of arguments for 'AI.DAGRUN' command
    try:
        command = "AI.DAGRUN "

        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("wrong number of arguments for 'AI.DAGRUN' command",exception.__str__())

    # ERR DAG with no ops
    command = "AI.TENSORSET volatile_tensor1 FLOAT 1 2 VALUES 5 10"
    ret = con.execute_command(command)
    env.assertEqual(ret, b'OK')
    try:
        command = "AI.DAGRUN LOAD 1 volatile_tensor1"
        ret = con.execute_command(command)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("DAG is empty",exception.__str__())


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
        env.assertEqual("unsupported command within DAG",exception.__str__())

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


def test_dagrun_ro_modelrun_scriptrun_resnet(env):
    if (not TEST_TF or not TEST_PT):
        return
    if(VALGRIND):
        env.debugPrint("skipping {} since it's hanging CI".format(sys._getframe().f_code.co_name), force=True)
        env.skip()

    con = env.getConnection()
    model_name = 'imagenet_model{{1}}'
    script_name = 'imagenet_script{{1}}'
    inputvar = 'images'
    outputvar = 'output'
    model_pb, script, labels, img = load_resnet_test_data()

    ret = con.execute_command('AI.MODELSET', model_name, 'TF', DEVICE,
                              'INPUTS', inputvar,
                              'OUTPUTS', outputvar,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.SCRIPTSET', script_name, DEVICE, 'SOURCE', script)
    env.assertEqual(ret, b'OK')
    #
    for opnumber in range(1,100):
        image_key = 'image{{1}}'
        temp_key1 = 'temp_key1{{1}}'
        temp_key2 = 'temp_key2{{1}}'
        class_key = 'output{{1}}'

        ret = con.execute_command(
            'AI.DAGRUN_RO', '|>',
            'AI.TENSORSET', image_key,
                'UINT8', img.shape[1], img.shape[0], 3,
                'BLOB', img.tobytes(),
                '|>',
            'AI.SCRIPTRUN',  script_name, 'pre_process_3ch',
                'INPUTS', image_key, 'OUTPUTS', temp_key1,
                '|>',
            'AI.MODELRUN', model_name,
                'INPUTS', temp_key1, 'OUTPUTS', temp_key2,
                '|>',
            'AI.SCRIPTRUN',  script_name, 'post_process',
                'INPUTS', temp_key2, 'OUTPUTS', class_key,
                '|>',
            'AI.TENSORGET', class_key, 'VALUES'
        )
        env.assertEqual([b'OK',b'OK',b'OK',b'OK'], ret[0:4])
        # tf model has 100 classes [0,999]
        env.assertEqual(ret[4][0] >= 0 and ret[4][0] < 1001, True)


def test_dagrun_modelrun_scriptrun_resnet(env):
    if (not TEST_TF or not TEST_PT):
        return
    if(VALGRIND):
        env.debugPrint("skipping {} since it's hanging CI".format(sys._getframe().f_code.co_name), force=True)
        env.skip()
    con = env.getConnection()
    model_name = 'imagenet_model:{{1}}'
    script_name = 'imagenet_script:{{1}}'
    image_key = 'image:{{1}}'
    temp_key1 = 'temp_key1'
    temp_key2 = 'temp_key2'
    class_key = 'output:{{1}}'
    inputvar = 'images'
    outputvar = 'output'
    model_pb, script, labels, img = load_resnet_test_data()

    ret = con.execute_command('AI.MODELSET', model_name, 'TF', DEVICE,
                              'INPUTS', inputvar,
                              'OUTPUTS', outputvar,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.SCRIPTSET', script_name, DEVICE, 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    #
    for opnumber in range(1,100):


        ret = con.execute_command(
            'AI.DAGRUN', 'PERSIST', '1', class_key, '|>',
            'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(), '|>',
            'AI.SCRIPTRUN',  script_name, 'pre_process_3ch',
                         'INPUTS', image_key,
                         'OUTPUTS', temp_key1,  '|>',
            'AI.MODELRUN', model_name,
                         'INPUTS', temp_key1,
                         'OUTPUTS', temp_key2,  '|>',
            'AI.SCRIPTRUN',  script_name, 'post_process',
                          'INPUTS', temp_key2,
                          'OUTPUTS', class_key
        )
        env.assertEqual([b'OK',b'OK',b'OK',b'OK'],ret)

        ret = con.execute_command('AI.TENSORGET', class_key, 'VALUES' )
        # tf model has 100 classes [0,999]
        env.assertEqual(ret[0]>=0 and ret[0]<1001, True)


def test_dag_scriptrun_errors(env):
    if (not TEST_TF or not TEST_PT):
        return
    con = env.getConnection()
    model_name = 'imagenet_model{{1}}'
    script_name = 'imagenet_script{{1}}'
    inputvar = 'images'
    outputvar = 'output'
    model_pb, script, labels, img = load_resnet_test_data()

    ret = con.execute_command('AI.MODELSET', model_name, 'TF', DEVICE,
                              'INPUTS', inputvar,
                              'OUTPUTS', outputvar,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ret = con.execute_command('AI.SCRIPTSET', script_name, DEVICE, 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    # ERR wrong number of inputs
    try:
        image_key = 'image{{1}}'
        temp_key1 = 'temp_key1{{1}}'
        temp_key2 = 'temp_key2{{1}}'
        class_key = 'output{{1}}'

        ret = con.execute_command(
            'AI.DAGRUN', '|>',
            'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(), '|>',
            'AI.SCRIPTRUN',  script_name,
            'INPUTS', image_key,
            'OUTPUTS', temp_key1, '|>',
            'AI.MODELRUN', model_name,
            'INPUTS', temp_key1,
            'OUTPUTS', temp_key2, '|>',
            'AI.SCRIPTRUN',  script_name, 'post_process',
            'INPUTS', temp_key2,
            'OUTPUTS', class_key
        )
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("function name not specified", exception.__str__())


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
        env.assertEqual("Number of keys given as INPUTS here does not match model definition",exception.__str__())


def test_dag_local_tensorset(env):
    con = env.getConnection()

    command = "AI.DAGRUN |> "\
        "AI.TENSORSET volatile_tensor1 FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORSET volatile_tensor2 FLOAT 1 2 VALUES 5 10 "

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK',b'OK'])

    # assert that transaction tensor does not exist
    ret = con.execute_command("EXISTS volatile_tensor")
    env.assertEqual(ret, 0 )


def test_dagro_local_tensorset(env):
    con = env.getConnection()

    command = "AI.DAGRUN_RO |> "\
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

    command = "AI.DAGRUN |> "\
        "AI.TENSORSET volatile_tensor FLOAT 1 2 VALUES 5 10 |> "\
        "AI.TENSORGET volatile_tensor META VALUES"

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK', [b'dtype', b'FLOAT', b'shape', [1, 2], b'values', [b'5', b'10']]])


def test_dag_keyspace_tensorget(env):
    con = env.getConnection()

    ret = con.execute_command(
        "AI.TENSORSET persisted_tensor FLOAT 1 2 VALUES 5 10")
    env.assertEqual(ret, b'OK')

    command = "AI.DAGRUN LOAD 1 persisted_tensor " \
              "|> "\
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


def test_dag_modelrun_financialNet_autobatch(env):
    if not TEST_TF:
        return
    con = env.getConnection()

    model_pb, creditcard_transactions, creditcard_referencedata = load_creditcardfraud_data(
        env)
    model_name = 'financialNet{{hhh}}'

    ret = con.execute_command('AI.MODELSET', model_name, 'TF', 'CPU',
                              'BATCHSIZE', 2, 'MINBATCHSIZE', 2,
                              'INPUTS', 'transaction', 'reference', 'OUTPUTS', 'output',
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
                    'AI.DAGRUN', 'LOAD', '1', reference_tensor_keyname, '|>',
                    'AI.TENSORSET', transaction_tensor_keyname, 'FLOAT', 1, 30,'BLOB', transaction_tensor.tobytes(), '|>',
                    'AI.MODELRUN', model_name,
                                'INPUTS', transaction_tensor_keyname, reference_tensor_keyname,
                                'OUTPUTS', result_tensor_keyname
                )
                ensureSlaveSynced(con, env)

            t = threading.Thread(target=run)
            t.start()

            ret = con.execute_command(
                'AI.DAGRUN', 'LOAD', '1', reference_tensor_keyname,
                            'PERSIST', '1', result_tensor_keyname, '|>',
                'AI.TENSORSET', transaction_tensor_keyname, 'FLOAT', 1, 30,'BLOB', transaction_tensor.tobytes(), '|>',
                'AI.MODELRUN', model_name,
                            'INPUTS', transaction_tensor_keyname, reference_tensor_keyname,
                            'OUTPUTS', result_tensor_keyname, '|>',
                'AI.TENSORGET', result_tensor_keyname, 'META',
            )

            ensureSlaveSynced(con, env)

            env.assertEqual([b'OK',b'OK',[b'dtype', b'FLOAT', b'shape', [1, 2]]], ret)

            # assert that transaction tensor does not exist
            ret = con.execute_command("EXISTS {}".format(transaction_tensor_keyname))
            env.assertEqual(ret, 0)
            # assert that result tensor exists
            ret = con.execute_command("EXISTS {}".format(result_tensor_keyname))
            env.assertEqual(ret, 1)
            t.join()


def test_dag_with_timeout(env):
    if not TEST_TF:
        return
    con = env.getConnection()
    batch_size = 2
    minbatch_size = 2
    timeout = 1
    model_name = 'model{1}'
    model_pb, input_var, output_var, labels, img = load_mobilenet_v2_test_data()

    con.execute_command('AI.MODELSET', model_name, 'TF', DEVICE,
                        'BATCHSIZE', batch_size, 'MINBATCHSIZE', minbatch_size,
                        'INPUTS', input_var,
                        'OUTPUTS', output_var,
                        'BLOB', model_pb)
    con.execute_command('AI.TENSORSET', 'input{1}',
                        'FLOAT', 1, img.shape[1], img.shape[0], img.shape[2],
                        'BLOB', img.tobytes())

    res = con.execute_command('AI.DAGRUN',
                        'LOAD', '1', 'input{1}', 
                        'TIMEOUT', timeout, '|>',
                        'AI.MODELRUN', model_name,
                        'INPUTS', 'input{1}', 'OUTPUTS', 'output{1}',
                        '|>',
                        'AI.MODELRUN', model_name,
                        'INPUTS', 'input{1}', 'OUTPUTS', 'output{1}')

    env.assertEqual(b'TIMEDOUT', res)


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


def test_dagrun_modelrun_multidevice_resnet(env):
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

    ret = con.execute_command('AI.MODELSET', model_name_0, 'TF', device_0,
                              'INPUTS', inputvar,
                              'OUTPUTS', outputvar,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELSET', model_name_1, 'TF', device_1,
                              'INPUTS', inputvar,
                              'OUTPUTS', outputvar,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.SCRIPTSET', script_name, device_0, 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)
 
    try:
        ret = con.execute_command(
            'AI.DAGRUN', '|>',
            'AI.SCRIPTRUN',  script_name, 'pre_process_3ch',
                         'INPUTS', image_key,
                         'OUTPUTS', temp_key1,
        )
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("INPUT key cannot be found in DAG", exception.__str__())

    try:
        ret = con.execute_command(
            'AI.DAGRUN', '|>',
            'AI.MODELRUN', model_name_0,
                         'INPUTS', temp_key1,
                         'OUTPUTS', temp_key2_0,
        )
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("INPUT key cannot be found in DAG", exception.__str__())

    try:
        ret = con.execute_command(
            'AI.DAGRUN', '|>',
            'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(),
                         '|>',
            'AI.SCRIPTRUN',  script_name, 'wrong_fn',
                         'INPUTS', image_key,
                         'OUTPUTS', temp_key1,
        )
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertTrue(exception.__str__().startswith("attempted to get undefined function wrong_fn"))

    try:
        ret = con.execute_command(
            'AI.DAGRUN', '|>',
            'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(),
                         '|>',
            'AI.SCRIPTRUN',  script_name, 'pre_process_3ch',
                         'INPUTS', image_key,
                         'OUTPUTS', temp_key1,
                         '|>',
            'AI.MODELRUN', model_name_0,
                         'INPUTS', temp_key1, temp_key1,
                         'OUTPUTS', temp_key2_0,
        )
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Number of keys given as INPUTS here does not match model definition", exception.__str__())

    ret = con.execute_command(
        'AI.DAGRUN',
                     'PERSIST', '2', class_key_0, class_key_1, '|>',
        'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(),
                     '|>',
        'AI.SCRIPTRUN',  script_name, 'pre_process_3ch',
                     'INPUTS', image_key,
                     'OUTPUTS', temp_key1,
                     '|>',
        'AI.MODELRUN', model_name_0,
                     'INPUTS', temp_key1,
                     'OUTPUTS', temp_key2_0,
                     '|>',
        'AI.MODELRUN', model_name_1,
                     'INPUTS', temp_key1,
                     'OUTPUTS', temp_key2_1,
                     '|>',
        'AI.SCRIPTRUN', script_name, 'post_process',
                      'INPUTS', temp_key2_0,
                      'OUTPUTS', class_key_0,
                      '|>',
        'AI.SCRIPTRUN', script_name, 'post_process',
                      'INPUTS', temp_key2_1,
                      'OUTPUTS', class_key_1
    )
    env.assertEqual([b'OK', b'OK', b'OK', b'OK', b'OK', b'OK'], ret)

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.TENSORGET', class_key_0, 'VALUES' )
    # tf model has 100 classes [0,999]
    env.assertEqual(ret[0]>=0 and ret[0]<1001, True)

    ret = con.execute_command('AI.TENSORGET', class_key_1, 'VALUES' )
    env.assertEqual(ret[0]>=0 and ret[0]<1001, True)


def test_dagrun_modelrun_multidevice_resnet_ensemble_alias(env):
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

    ret = con.execute_command('AI.MODELSET', model_name_0, 'TF', device_0,
                              'INPUTS', inputvar,
                              'OUTPUTS', outputvar,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELSET', model_name_1, 'TF', device_1,
                              'INPUTS', inputvar,
                              'OUTPUTS', outputvar,
                              'BLOB', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.SCRIPTSET', script_name_0, device_0, 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.SCRIPTSET', script_name_1, device_1, 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    try:
        ret = con.execute_command(
            'AI.DAGRUN',
                         'PERSIST', '2', class_key_0, class_key_1, '|>',
            'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(),
                         '|>',
            'AI.SCRIPTRUN',  script_name_0, 'pre_process_3ch',
                         'INPUTS', image_key,
                         'OUTPUTS', temp_key1,
                         '|>',
            'AI.MODELRUN', model_name_0,
                         'INPUTS', temp_key1,
                         'OUTPUTS', temp_key2_0,
                         '|>',
            'AI.MODELRUN', model_name_1,
                         'INPUTS', temp_key1,
                         'OUTPUTS', temp_key2_1,
                         '|>',
            'AI.SCRIPTRUN', script_name_1, 'ensemble',
                          'INPUTS', temp_key2_0, temp_key2_1,
                          'OUTPUTS', temp_key1,
                          '|>',
            'AI.SCRIPTRUN', script_name_0, 'post_process',
                          'INPUTS', temp_key1,
                          'OUTPUTS', class_key_0,
        )
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("PERSIST key cannot be found in DAG", exception.__str__())

    try:
        ret = con.execute_command(
            'AI.DAGRUN',
                         'PERSIST', '1', class_key_0,
                         '|>',
            'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(),
                         '|>',
            'AI.SCRIPTRUN',  script_name_0, 'pre_process_3ch',
                         'INPUTS', image_key,
                         'OUTPUTS', temp_key1,
                         '|>',
            'AI.MODELRUN', model_name_0,
                         'INPUTS', temp_key1 + '_foo',
                         'OUTPUTS', temp_key2_0,
                         '|>',
            'AI.MODELRUN', model_name_1,
                         'INPUTS', temp_key1,
                         'OUTPUTS', temp_key2_1,
                         '|>',
            'AI.SCRIPTRUN', script_name_1, 'ensemble',
                          'INPUTS', temp_key2_0, temp_key2_1,
                          'OUTPUTS', temp_key1,
                          '|>',
            'AI.SCRIPTRUN', script_name_0, 'post_process',
                          'INPUTS', temp_key1,
                          'OUTPUTS', class_key_0,
        )
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("INPUT key cannot be found in DAG", exception.__str__())

    try:
        ret = con.execute_command(
            'AI.DAGRUN',
                         'PERSIST', '1', class_key_0, '|>',
            'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(),
                         '|>',
            'AI.SCRIPTRUN',  script_name_0, 'pre_process_3ch',
                         'INPUTS', image_key,
                         'OUTPUTS', temp_key1,
                         '|>',
            'AI.MODELRUN', model_name_0,
                         'INPUTS', temp_key1,
                         'OUTPUTS', temp_key2_0,
                         '|>',
            'AI.MODELRUN', model_name_1,
                         'INPUTS', temp_key1,
                         'OUTPUTS', temp_key2_1,
                         '|>',
            'AI.SCRIPTRUN', script_name_0, 'ensemble',
                          'INPUTS', temp_key2_0,
                          'OUTPUTS', temp_key1,
                          '|>',
            'AI.SCRIPTRUN', script_name_0, 'post_process',
                          'INPUTS', temp_key1,
                          'OUTPUTS', class_key_0,
        )
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertTrue(exception.__str__().startswith("expected 2 inputs, but got only 1"))

    ret = con.execute_command(
        'AI.DAGRUN',
                     'PERSIST', '1', class_key_0,
                     '|>',
        'AI.TENSORSET', image_key, 'UINT8', img.shape[1], img.shape[0], 3, 'BLOB', img.tobytes(),
                     '|>',
        'AI.SCRIPTRUN',  script_name_0, 'pre_process_3ch',
                     'INPUTS', image_key,
                     'OUTPUTS', temp_key1,
                      '|>',
        'AI.MODELRUN', model_name_0,
                     'INPUTS', temp_key1,
                     'OUTPUTS', temp_key2_0,
                      '|>',
        'AI.MODELRUN', model_name_1,
                     'INPUTS', temp_key1,
                     'OUTPUTS', temp_key2_1,
                     '|>',
        'AI.SCRIPTRUN', script_name_0, 'ensemble',
                      'INPUTS', temp_key2_0, temp_key2_1,
                      'OUTPUTS', temp_key1,
                      '|>',
        'AI.SCRIPTRUN', script_name_0, 'post_process',
                      'INPUTS', temp_key1,
                      'OUTPUTS', class_key_0,
    )
    env.assertEqual([b'OK', b'OK', b'OK', b'OK', b'OK', b'OK'], ret)

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.TENSORGET', class_key_0, 'VALUES' )
    # tf model has 100 classes [0,999]
    env.assertEqual(ret[0]>=0 and ret[0]<1001, True)
