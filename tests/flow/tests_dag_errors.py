from includes import *

'''
python -m RLTest --test tests_dag_errors.py --module path/to/redisai.so
'''

def test_dag_load_errors(env):
    con = get_connection(env, '{1}')

    # ERR wrong number of arguments for LOAD
    check_error_message(env, con, "missing arguments after LOAD keyword in DAG command",
                        "AI.DAGEXECUTE PERSIST 2 no_tensor1{1} no_tensor2{1} LOAD 1")

    # ERR invalid or negative number of arguments for LOAD
    check_error_message(env, con, "invalid or negative value found in number of keys to LOAD",
                        "AI.DAGEXECUTE LOAD notnumber{1} |> AI.TENSORGET no_tensor{1} VALUES")

    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    # ERR number of keys to LOAD does not match the number of given arguments.
    check_error_message(env, con, "number of keys to LOAD into DAG does not match the number of given arguments",
                        "AI.DAGEXECUTE ROUTING a{1} LOAD 2 a{1}")

    # ERR Check key in shard succeeded  but tensor is empty.
    check_error_message(env, con, "tensor key is empty or in a different shard",
                        "AI.DAGEXECUTE ROUTING no_tensor{1} LOAD 1 no_tensor{1}")

    # WRONGTYPE Operation against a key holding the wrong kind of value
    con.execute_command('SET', 'no_tensor{1}', 'value')
    check_error_message(env, con, "WRONGTYPE Operation against a key holding the wrong kind of value",
                        "AI.DAGEXECUTE ROUTING no_tensor{1} LOAD 1 no_tensor{1}")


def test_dag_persist_errors(env):
    con = get_connection(env, '{1}')
    con.execute_command('AI.TENSORSET', 'a{1}', 'FLOAT', 2, 2, 'VALUES', 2, 3, 2, 3)

    # ERR wrong number of arguments for PERSIST
    check_error_message(env, con, "missing arguments after PERSIST keyword in DAG command",
                        "AI.DAGEXECUTE ROUTING a{1} LOAD 1 a{1} PERSIST 1")

    # ERR invalid or negative value found in number of keys to PERSIST
    check_error_message(env, con, "invalid or negative value found in number of keys to PERSIST",
                        "AI.DAGEXECUTE ROUTING a{1} PERSIST not_number a{1}")

    # ERR number of keys to PERSIST does not match the number of given arguments.
    check_error_message(env, con, "number of keys to PERSIST after DAG execution does not match the number of given arguments",
                        "AI.DAGEXECUTE ROUTING a{1} PERSIST 2 a{1}")


def test_dag_timeout_errors(env):
    con = get_connection(env, '{1}')

    # ERR no value provided for timeout
    check_error_message(env, con, "No value provided for TIMEOUT",
                        "AI.DAGEXECUTE ROUTING a{1} PERSIST 1 a{1} TIMEOUT")

    # ERR invalid timeout value
    check_error_message(env, con, "Invalid value for TIMEOUT",
                        "AI.DAGEXECUTE ROUTING a{1} PERSIST 1 a{1} TIMEOUT not_number")


def test_dag_common_errors(env):
    con = get_connection(env, '{1}')

    model_pb = load_file_content('graph.pb')
    ret = con.execute_command('AI.MODELSTORE', 'm{1}', 'TF', DEVICE,
                              'INPUTS', 2, 'a', 'b', 'OUTPUTS', 1, 'mul', 'BLOB', model_pb)
    env.assertEqual(ret, b'OK')
    script = load_file_content('script.txt')
    ret = con.execute_command('AI.SCRIPTSTORE', 'script{1}', DEVICE, 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
    env.assertEqual(ret, b'OK')


    check_error_message(env, con, "Missing ROUTING value",
                        "AI.DAGEXECUTE PERSIST 3 a{1} b{1} c{1} ROUTING")

    # ERR bad syntax
    check_error_message(env, con, "Invalid DAG command. Unexpected argument:  BAD_ARG",
                        "AI.DAGEXECUTE ROUTING a{1} PERSIST 1 a{1} BAD_ARG")

    # ERR DAG doesn't contains none of ROUTING, LOAD, PERSIST
    check_error_message(env, con, "AI.DAGEXECUTE and AI.DAGEXECUTE_RO commands must contain at least one out of"
                                  " ROUTING, LOAD, PERSIST keywords",
                        "AI.DAGEXECUTE |> AI.TENSORSET a{1} FLOAT 1 2 VALUES 5 10")

    # ERR unsupported command within DAG
    check_error_message(env, con, "Unsupported command within DAG",
                        "AI.DAGEXECUTE ROUTING a{1} |> AI.NOCOMMAND a{1} FLOAT 1 2 VALUES 5 10")

    # ERR wrong number of arguments for 'AI.DAGEXECUTE' command
    check_error_message(env, con, "missing arguments for 'AI.DAGEXECUTE' command", "AI.DAGEXECUTE ")

    # ERR DAG with no ops
    command = "AI.TENSORSET volatile_tensor{1} FLOAT 2 2 VALUES 5 10 5 10"
    ret = con.execute_command(command)
    env.assertEqual(ret, b'OK')
    check_error_message(env, con, "DAG is empty", "AI.DAGEXECUTE ROUTING volatile_tensor{1} LOAD 1 volatile_tensor{1}")

    # ERR Deprecated commands in (non-deprecated) AI.DAGEXECUTE
    check_error_message(env, con, "Deprecated AI.MODELRUN cannot be used in AI.DAGEXECUTE command",
                        "AI.DAGEXECUTE LOAD 1 volatile_tensor{1} "
                        "|> AI.MODELRUN m{1} INPUTS volatile_tensor{1} volatile_tensor{1} OUTPUTS output_tensor{1}")

    check_error_message(env, con, "Deprecated AI.SCRIPTRUN cannot be used in AI.DAGEXECUTE command",
                        "AI.DAGEXECUTE LOAD 1 volatile_tensor{1} "
                        "|> AI.SCRIPTRUN script{1} bar INPUTS volatile_tensor{1} volatile_tensor{1} OUTPUTS output_tensor{1}")


def test_dag_ro_errors(env):
    con = get_connection(env, '{1}')

    # ERR wrong number of arguments for 'AI.DAGEXECUTE_RO' command
    check_error_message(env, con, "missing arguments for 'AI.DAGEXECUTE_RO' command", "AI.DAGEXECUTE_RO ")

    # ERR persist in not allowed in AI.DAGEXECUTE_RO
    check_error_message(env, con, "PERSIST cannot be specified in a read-only DAG",
                        "AI.DAGEXECUTE_RO PERSIST 1 tensor1{1} |> AI.TENSORSET tensor1{1} FLOAT 1 2 VALUES 5 10")

    # ERR AI.SCRIPTEXECUTE is not allowed in AI.DAGEXECUTE_RO
    script = load_file_content('script.txt')
    ret = con.execute_command('AI.SCRIPTSTORE', 'script{1}', DEVICE, 'ENTRY_POINTS', 2, 'bar', 'bar_variadic', 'SOURCE', script)
    env.assertEqual(ret, b'OK')
    ret = con.execute_command("AI.TENSORSET volatile_tensor{1} FLOAT 1 2 VALUES 5 10")
    env.assertEqual(ret, b'OK')

    check_error_message(env, con, "AI.SCRIPTEXECUTE command cannot be specified in a read-only DAG",
                        "AI.DAGEXECUTE_RO LOAD 1 volatile_tensor{1}"
                        " |> AI.SCRIPTEXECUTE script{1} bar INPUTS 2 volatile_tensor{1} volatile_tensor{1}")


def test_dag_scriptexecute_errors(env):
    if (not TEST_TF or not TEST_PT):
        return

    con = get_connection(env, '{1}')
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

    ret = con.execute_command('AI.SCRIPTSTORE', script_name, DEVICE, 'ENTRY_POINTS', 4, 'pre_process_3ch', 'pre_process_4ch', 'post_process', 'ensemble', 'SOURCE', script)
    env.assertEqual(ret, b'OK')

    # The function name in AI.SCRIPTEXECUTE is missing, so 'INPUTS' is considered as the function name, and
    # error is raised for the unexpected argument ("1") coming where INPUTS should have come.
    image_key = 'image{1}'
    temp_key1 = 'temp_key1{1}'
    temp_key2 = 'temp_key2{1}'
    class_key = 'output{1}'
    command = (
        'AI.DAGEXECUTE', 'ROUTING', '{1}', '|>',
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
    check_error_message(env, con, "Invalid AI.SCRIPTEXECUTE command. Unexpected argument: 1", *command)


def test_dag_modelexecute_financialNet_errors(env):
    if not TEST_TF:
        return
    con = get_connection(env, '{1}')
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


def test_dag_crossslot_violation_errors(env):

    if env.isCluster():
        con = get_connection(env, '{1}')

        # ERR CROSSSLOT violation (LOAD and PERSIST tensors has different hash tags)
        command = (
            'AI.DAGEXECUTE', 'LOAD', '1', 'referenceTensor:{1}',
            'PERSIST', '1', 'resultTensor:{2}', '|>',
            'AI.TENSORSET', 'resultTensor:{2}', 'FLOAT', 1, 30,
        )
        check_error_message(env, con, "Keys in request don't hash to the same slot", *command, error_type=redis.exceptions.ClusterCrossSlotError)

        # ERR CROSSSLOT violation (model key has a different hash tag than the LOAD and PERSIST tensors)
        command = (
            'AI.DAGEXECUTE', 'LOAD', '1', 'referenceTensor:{1}',
            'PERSIST', '1', 'resultTensor:{1}', '|>',
            'AI.TENSORSET', 'transactionTensor:{1}', 'FLOAT', 1, 30, '|>',
            'AI.MODELEXECUTE', 'model_key{2}',
            'INPUTS', 1, 'transactionTensor:{1}',
            'OUTPUTS', 1, 'resultTensor:{1}',
        )
        check_error_message(env, con, "Keys in request don't hash to the same slot", *command, error_type=redis.exceptions.ClusterCrossSlotError)

        command = (
            'AI.DAGEXECUTE', 'LOAD', '1', 'referenceTensor:{1}',
            'PERSIST', '1', 'resultTensor:{1}', '|>',
            'ROUTING', 'model_key{2}',
            'AI.TENSORSET', 'transactionTensor:{1}', 'FLOAT', 1, 30, '|>',
            'AI.MODELEXECUTE', 'model_key{2}',
            'INPUTS', 1, 'transactionTensor:{1}',
            'OUTPUTS', 1, 'resultTensor:{1}',
        )
        check_error_message(env, con, "Keys in request don't hash to the same slot", *command, error_type=redis.exceptions.ClusterCrossSlotError)


def test_dag_tensorget_tensorset_errors(env):
    con = get_connection(env, '{1}')

    # ERR insufficient args for tensor get
    check_error_message(env, con, "wrong number of arguments for 'AI.TENSORGET' command",
                        "AI.DAGEXECUTE ROUTING a{1} |> AI.TENSORSET a{1} FLOAT 1 |> AI.TENSORGET")

    # ERR too many args for tensor get
    check_error_message(env, con, "wrong number of arguments for 'AI.TENSORGET' command",
                        "AI.DAGEXECUTE ROUTING a{1} |> AI.TENSORSET a{1} FLOAT 1 |> AI.TENSORGET a{1} META BLOB extra")

    # ERR insufficient args for tensor set
    check_error_message(env, con, "wrong number of arguments for 'AI.TENSORSET' command",
                        "AI.DAGEXECUTE ROUTING a{1} |> AI.TENSORSET a{1} FLOAT 1 |> AI.TENSORSET")


def test_dag_error_before_tensorget_op(env):
    if not TEST_TF:
        return

    con = get_connection(env, '{1}')
    tf_model = load_file_content('graph.pb')
    ret = con.execute_command('AI.MODELSTORE', 'tf_model{1}', 'TF', DEVICE,
                              'INPUTS', 2, 'a', 'b',
                              'OUTPUTS', 1, 'mul',
                              'BLOB', tf_model)
    env.assertEqual(b'OK', ret)

    # Run the model from DAG context, where MODELEXECUTE op fails due to dim mismatch in one of the tensors inputs:
    # the input tensor 'b' is considered as tensor with dim 2X2X3 initialized with zeros, while the model expects that
    # both inputs to node 'mul' will be with dim 2.
    ret = con.execute_command('AI.DAGEXECUTE_RO', 'ROUTING', '{1}',
                              '|>', 'AI.TENSORSET', 'a', 'FLOAT', 2, 'VALUES', 2, 3,
                              '|>', 'AI.TENSORSET', 'b', 'FLOAT', 2, 2, 3,
                              '|>', 'AI.MODELEXECUTE', 'tf_model{1}', 'INPUTS', 2, 'a', 'b', 'OUTPUTS', 1, 'tD',
                              '|>', 'AI.TENSORGET', 'tD', 'VALUES')

    # Expect that the MODELEXECUTE op will raise an error, and the last TENSORGET op will not be executed
    env.assertEqual(ret[0], b'OK')
    env.assertEqual(ret[1], b'OK')
    env.assertEqual(ret[3], b'NA')
    env.assertEqual(type(ret[2]), redis.exceptions.ResponseError)
    expected_error_msg = 'Incompatible shapes: [2] vs. [2,2,3] \t [[{{node mul}}]]'
    if DEVICE != 'CPU':
        expected_error_msg = 'Invalid argument: required broadcastable shapes' \
                             ' 	 [[{{node mul}}]]'
    env.assertContains(expected_error_msg, str(ret[2]))
