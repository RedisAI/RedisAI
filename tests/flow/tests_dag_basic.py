from includes import *

'''
python -m RLTest --test tests_dag_basic.py --module path/to/redisai.so
'''


def test_dag_load(env):
    con = env.getConnection()
    ret = con.execute_command(
        "AI.TENSORSET persisted_tensor_1{1} FLOAT 1 2 VALUES 5 10")
    env.assertEqual(ret, b'OK')
    command = "AI.DAGEXECUTE LOAD 1 persisted_tensor_1{1}" \
              " |> AI.TENSORGET persisted_tensor_1{1} VALUES"
    ret = con.execute_command(command)
    env.assertEqual(ret[0], [b'5', b'10'])


def test_dag_local_tensorset(env):
    con = env.getConnection()

    command = "AI.DAGEXECUTE KEYS 1 {1} |> " \
              "AI.TENSORSET volatile_tensor1 FLOAT 1 2 VALUES 5 10 |> " \
              "AI.TENSORSET volatile_tensor2 FLOAT 1 2 VALUES 5 10 "

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK',b'OK'])

    # assert that transaction tensor does not exist
    ret = con.execute_command("EXISTS volatile_tensor")
    env.assertEqual(ret, 0)


def test_dagro_local_tensorset(env):
    con = env.getConnection()

    command = "AI.DAGEXECUTE_RO KEYS 1 {some_tag} |> " \
              "AI.TENSORSET volatile_tensor1 FLOAT 1 2 VALUES 5 10 |> " \
              "AI.TENSORSET volatile_tensor2 FLOAT 1 2 VALUES 5 10 "

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK',b'OK'])

    # assert that volatile_tensor does not exist
    ret = con.execute_command("EXISTS volatile_tensor")
    env.assertEqual(ret, 0 )


def test_dag_local_tensorset_persist(env):
    con = env.getConnection()

    command = "AI.DAGEXECUTE " \
              "PERSIST 1 tensor1{1} |> " \
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

    command = "AI.DAGEXECUTE " \
              "PERSIST 1 tensor3:{1} |> " \
              "AI.TENSORSET tensor1{1} FLOAT 1 2 VALUES 5 10 |> " \
              "AI.TENSORSET tensor2 FLOAT 1 2 VALUES 5 10 |> " \
              "AI.TENSORSET tensor3:{1} FLOAT 1 2 VALUES 5 10 |> " \
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

    command = "AI.DAGEXECUTE PERSIST 1 tensor1{1} |> " \
              "AI.TENSORSET tensor1{1} FLOAT 1 2 VALUES 5 10 |> " \
              "AI.TENSORGET tensor1{1} VALUES"

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK', [b'5', b'10']])

    ret = con.execute_command("AI.TENSORGET tensor1{1} VALUES")
    env.assertEqual(ret, [b'5', b'10'])


def test_dag_local_multiple_tensorset_on_same_tensor(env):
    con = env.getConnection()

    command = "AI.DAGEXECUTE PERSIST 1 tensor1{1} |> " \
              "AI.TENSORSET tensor1{1} FLOAT 1 2 VALUES 5 10 |> " \
              "AI.TENSORGET tensor1{1} META VALUES |> " \
              "AI.TENSORSET tensor1{1} FLOAT 1 4 VALUES 20 40 60 80 |> " \
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
              " PERSIST 1 volatile_tensor_persisted{1} |> " \
              "AI.TENSORSET volatile_tensor_persisted{1} FLOAT 1 2 VALUES 5 10 |> " \
              "AI.TENSORGET persisted_tensor_1{1} META VALUES |> " \
              "AI.TENSORGET persisted_tensor_2:{1} META VALUES "

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK', [b'dtype', b'FLOAT', b'shape', [1, 2], b'values', [b'5', b'10']], [
        b'dtype', b'FLOAT', b'shape', [1, 3], b'values', [b'0', b'0', b'0']]])

    ret = con.execute_command("AI.TENSORGET volatile_tensor_persisted{1} META VALUES")
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

    command = "AI.DAGEXECUTE_RO LOAD 1 persisted_tensor |> " \
              "AI.TENSORGET persisted_tensor VALUES"

    ret = con.execute_command(command)
    env.assertEqual(ret, [[b'5', b'10']])


def test_dag_keyspace_and_localcontext_tensorget(env):
    con = env.getConnection()

    ret = con.execute_command(
        "AI.TENSORSET persisted_tensor FLOAT 1 2 VALUES 5 10")
    env.assertEqual(ret, b'OK')

    command = "AI.DAGEXECUTE LOAD 1 persisted_tensor |> " \
              "AI.TENSORSET volatile_tensor FLOAT 1 2 VALUES 5 10 |> " \
              "AI.TENSORGET persisted_tensor VALUES |> " \
              "AI.TENSORGET volatile_tensor VALUES"

    ret = con.execute_command(command)
    env.assertEqual(ret, [b'OK', [b'5', b'10'], [b'5', b'10']])


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
