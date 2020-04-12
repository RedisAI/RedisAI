import redis

from includes import *

'''
python -m RLTest --test tests_dag.py --module path/to/redisai.so
'''


# def test_dag_local_tensorset(env):
#     con = env.getConnection()

#     command = "AI.DAGRUN "\
#         "AI.TENSORSET volative_tensor FLOAT 1 2 VALUES 5 10"

#     ret = con.execute_command(command)
#     env.assertEqual(ret, [b'OK'])


# def test_dag_local_tensorset_persist(env):
#     con = env.getConnection()

#     command = "AI.DAGRUN PERSIST 1 volative_tensor_persisted |> "\
#         "AI.TENSORSET volative_tensor_persisted FLOAT 1 2 VALUES 5 10 |> "\
#         "AI.TENSORGET volative_tensor_persisted VALUES"

#     ret = con.execute_command(command)
#     env.assertEqual(ret, [b'OK', [b'FLOAT', [1, 2], [b'5', b'10']]])

#     ret = con.execute_command("AI.TENSORGET volative_tensor_persisted VALUES")
#     env.assertEqual(ret, [b'FLOAT', [1, 2], [b'5', b'10']])


# def test_dag_load_persist_tensorset_tensorget(env):
#     con = env.getConnection()

#     ret = con.execute_command(
#         "AI.TENSORSET persisted_tensor_1 FLOAT 1 2 VALUES 5 10")
#     env.assertEqual(ret, b'OK')

#     ret = con.execute_command(
#         "AI.TENSORSET persisted_tensor_2 FLOAT 1 3 VALUES 0 0 0")
#     env.assertEqual(ret, b'OK')

#     command = "AI.DAGRUN LOAD 2 persisted_tensor_1 persisted_tensor_2 PERSIST 1 volative_tensor_persisted |> "\
#         "AI.TENSORSET volative_tensor_persisted FLOAT 1 2 VALUES 5 10 |> "\
#         "AI.TENSORGET persisted_tensor_1 VALUES |> "\
#         "AI.TENSORGET persisted_tensor_2 VALUES "

#     ret = con.execute_command(command)
#     env.assertEqual(ret, [b'OK', [b'FLOAT', [1, 2], [b'5', b'10']], [
#                     b'FLOAT', [1, 3], [b'0', b'0', b'0']]])

#     ret = con.execute_command("AI.TENSORGET volative_tensor_persisted VALUES")
#     env.assertEqual(ret, [b'FLOAT', [1, 2], [b'5', b'10']])


# def test_dag_local_tensorset_tensorget(env):
#     con = env.getConnection()

#     command = "AI.DAGRUN "\
#         "AI.TENSORSET volative_tensor FLOAT 1 2 VALUES 5 10 |> "\
#         "AI.TENSORGET volative_tensor VALUES"

#     ret = con.execute_command(command)
#     env.assertEqual(ret, [b'OK', [b'FLOAT', [1, 2], [b'5', b'10']]])


# def test_dag_keyspace_tensorget(env):
#     con = env.getConnection()

#     ret = con.execute_command(
#         "AI.TENSORSET persisted_tensor FLOAT 1 2 VALUES 5 10")
#     env.assertEqual(ret, b'OK')

#     command = "AI.DAGRUN LOAD 1 persisted_tensor |> "\
#         "AI.TENSORGET persisted_tensor VALUES"

#     ret = con.execute_command(command)
#     env.assertEqual(ret, [[b'FLOAT', [1, 2], [b'5', b'10']]])


# def test_dag_keyspace_and_localcontext_tensorget(env):
#     con = env.getConnection()

#     ret = con.execute_command(
#         "AI.TENSORSET persisted_tensor FLOAT 1 2 VALUES 5 10")
#     env.assertEqual(ret, b'OK')

#     command = "AI.DAGRUN LOAD 1 persisted_tensor |> "\
#         "AI.TENSORSET volative_tensor FLOAT 1 2 VALUES 5 10 |> "\
#         "AI.TENSORGET persisted_tensor VALUES |> "\
#         "AI.TENSORGET volative_tensor VALUES"

#     ret = con.execute_command(command)
#     env.assertEqual(ret, [b'OK', [b'FLOAT', [1, 2], [b'5', b'10']], [
#                     b'FLOAT', [1, 2], [b'5', b'10']]])


def test_dag_modelrun_financialNet(env):
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
    for transaction_tensor in creditcard_transactions:
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
        env.assertEqual(ret, [b'FLOAT', [1, 2]])

        # assert that transaction tensor does not exist
        ret = con.execute_command("EXISTS transactionTensor:{} META".format(
            tensor_number))
        env.assertEqual(ret, 0 )
        tensor_number = tensor_number + 1


        

