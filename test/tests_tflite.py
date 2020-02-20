import redis

from includes import *

'''
python -m RLTest --test tests_common.py --module path/to/redisai.so
'''


def test_run_tflite_model(env):
    if not TEST_TFLITE:
        return

    con = env.getConnection()

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    model_filename = os.path.join(test_data_path, 'mnist_model_quant.tflite')
    wrong_model_filename = os.path.join(test_data_path, 'graph.pb')
    sample_filename = os.path.join(test_data_path, 'one.raw')

    with open(model_filename, 'rb') as f:
        model_pb = f.read()

    with open(model_filename, 'rb') as f:
        model_pb2 = f.read()

    with open(wrong_model_filename, 'rb') as f:
        wrong_model_pb = f.read()

    with open(sample_filename, 'rb') as f:
        sample_raw = f.read()

    ret = con.execute_command('AI.MODELSET', 'm', 'TFLITE', 'CPU', model_pb)
    env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    ret = con.execute_command('AI.MODELGET', 'm')
    env.assertEqual(len(ret), 3)
    # TODO: enable me
    # env.assertEqual(ret[0], b'TFLITE')
    # env.assertEqual(ret[1], b'CPU')

    try:
        con.execute_command('AI.MODELSET', 'm_1', 'TFLITE', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    ret = con.execute_command('AI.MODELSET', 'm_2', 'TFLITE', 'CPU', model_pb2)

    try:
        con.execute_command('AI.MODELSET', 'm_2', model_pb)
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    con.execute_command('AI.TENSORSET', 'a', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)

    try:
        con.execute_command('AI.MODELRUN', 'm_2', 'INPUTS', 'a', 'OUTPUTS')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm_2', 'INPUTS', 'a', 'b', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm_2', 'a', 'b', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm_2', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm', 'OUTPUTS', 'c')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'b')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'OUTPUTS')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    try:
        con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'OUTPUTS', 'b')
    except Exception as e:
        exception = e
    env.assertEqual(type(exception), redis.exceptions.ResponseError)

    con.execute_command('AI.MODELRUN', 'm', 'INPUTS', 'a', 'OUTPUTS', 'b', 'c')

    ensureSlaveSynced(con, env)

    tensor = con.execute_command('AI.TENSORGET', 'b', 'VALUES')
    value = tensor[-1][0]

    env.assertEqual(value, 1)

    
