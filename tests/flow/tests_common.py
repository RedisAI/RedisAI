import redis

from includes import *

'''
python -m RLTest --test tests_common.py --module path/to/redisai.so
'''


def test_common_tensorset(env):
    con = get_connection(env, '{0}')

    tested_datatypes = ["FLOAT", "DOUBLE", "INT8", "INT16", "INT32", "INT64", "UINT8", "UINT16", "BOOL"]
    for datatype in tested_datatypes:
        ret = con.execute_command('AI.TENSORSET', 'tensor_{}{{0}}'.format(datatype), datatype, 2, 'VALUES', 1, 1)
        env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    # AI.TENSORGET in BLOB format and set in a new key
    for datatype in tested_datatypes:
        _, tensor_dtype, _, tensor_dim, _, tensor_blob = con.execute_command('AI.TENSORGET', 'tensor_{}{{0}}'.format(datatype),
                                                                             'META', 'BLOB')
        ret = con.execute_command('AI.TENSORSET', 'tensor_blob_{}{{0}}'.format(datatype), datatype, 2, 'BLOB', tensor_blob)
        env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    reply_types = ["META", "VALUES", "BLOB"]
    # Confirm that tensor_{0} and tensor_blob_{0} are equal for META VALUES BLOB
    for datatype in tested_datatypes:
        for reply_type in reply_types:
            tensor_1_reply = con.execute_command('AI.TENSORGET', 'tensor_{}{{0}}'.format(datatype), reply_type)
            tensor_2_reply = con.execute_command('AI.TENSORGET', 'tensor_blob_{}{{0}}'.format(datatype), reply_type)
            env.assertEqual(tensor_1_reply, tensor_2_reply)


def test_common_tensorset_error_replies(env):
    con = get_connection(env, '{0}')
    sample_raw = load_file_content('one.raw')

    ret = con.execute_command('AI.TENSORSET', 'sample_raw_ok{0}', 'FLOAT', 1, 1, 28, 28, 'BLOB', sample_raw)
    env.assertEqual(ret, b'OK')

    # WRONGTYPE Operation against a key holding the wrong kind of value
    con.execute_command('SET', 'non-tensor{0}', 'value')
    check_error_message(env, con, "WRONGTYPE Operation against a key holding the wrong kind of value",
                        'AI.TENSORSET', 'non-tensor{0}', 'INT32', 2, 'unsupported', 2, 3)

    # ERR unsupported tensor type
    check_error_message(env, con, "invalid data type",
                        'AI.TENSORSET', 'z{0}', 'INT128', 2, 'VALUES', 2, 3)

    # ERR negative value found in tensor shape
    check_error_message(env, con, "invalid or negative value found in tensor shape",
                        'AI.TENSORSET', 'z{0}', 'INT32', -1, 'VALUES', 2, 3)

    # ERR invalid argument found in tensor shape
    check_error_message(env, con, "invalid or negative value found in tensor shape",
                        'AI.TENSORSET', 'z{0}', 'INT32', 2, 'not_a_number', 'VALUES', 2, 3)

    # ERR invalid value
    check_error_message(env, con, "invalid value",
                        'AI.TENSORSET', 'z{0}', 'FLOAT', 2, 'VALUES', 2, 'A')

    # ERR invalid value
    check_error_message(env, con, "invalid value",
                        'AI.TENSORSET', 'z{0}', 'INT32', 2, 'VALUES', 2, '1.87')

    # ERR invalid value - overflow
    check_error_message(env, con, "invalid value",
                        'AI.TENSORSET', 'z{0}', 'BOOL', 2, 'VALUES', 1, 2)

    # ERR invalid string value
    check_error_message(env, con, "C-string value contains null character",
                        'AI.TENSORSET', 'z{0}', 'STRING', 2, 'VALUES', 'valid C-string', 'invalid\0C-string')

    # ERR invalid value - overflow
    check_error_message(env, con, "invalid value",
                        'AI.TENSORSET', 'z{0}', 'INT8', 2, 'VALUES', -1, -128)

    # ERR insufficient number of args
    check_error_message(env, con, "wrong number of arguments for 'AI.TENSORSET' command",
                        'AI.TENSORSET')

    # ERR insufficient number of args
    check_error_message(env, con, "wrong number of arguments for 'AI.TENSORSET' command",
                        'AI.TENSORSET', 'z{0}')

    # ERR insufficient number of args
    check_error_message(env, con, "wrong number of arguments for 'AI.TENSORSET' command",
                        'AI.TENSORSET', 'z{0}', 'FLOAT')

    # ERR insufficient number of values (expected 2)
    check_error_message(env, con, "wrong number of values was given in 'AI.TENSORSET' command",
                        'AI.TENSORSET', 'z{0}', 'FLOAT', 2, 'VALUES', 1)

    # ERR too many values (expected 2)
    check_error_message(env, con, "wrong number of values was given in 'AI.TENSORSET' command",
                        'AI.TENSORSET', 'z{0}', 'FLOAT', 2, 'VALUES', 1, 2, 3)

    # ERR in blob - extra argument
    check_error_message(env, con, "a single binary string should come after the BLOB argument in 'AI.TENSORSET' command",
                        'AI.TENSORSET', 'blob_tensor_more_args{0}', 'FLOAT', 2, 'BLOB', '\x00', 'extra-argument')

    # ERR in blob - missing argument
    check_error_message(env, con, "a single binary string should come after the BLOB argument in 'AI.TENSORSET' command",
                        'AI.TENSORSET', 'blob_tensor_less_args{0}', 'FLOAT', 2, 'BLOB')

    # ERR in blob - blob size is not compatible with tensor meta data
    check_error_message(env, con, "data length does not match tensor shape and type",
                        'AI.TENSORSET', 'sample_raw_wrong_blob_for_dim{0}', 'FLOAT', 1, 1, 28, 280, 'BLOB', sample_raw)

    # ERR in string tensor blob - number of strings is not compatible with tensor meta data
    check_error_message(env, con, "Number of string elements in data blob does not match tensor length",
                        'AI.TENSORSET', 'z{0}', 'STRING', 2, 'BLOB', 'single c-string\0')

    # ERR in string tensor blob - number of strings is not compatible with tensor meta data
    check_error_message(env, con, "Number of string elements in data blob does not match tensor length",
                        'AI.TENSORSET', 'z{0}', 'STRING', 2, 'BLOB', 'one\0two\0three\0')

    # ERR in string tensor blob - number of strings is not compatible with tensor meta data
    check_error_message(env, con, "Number of string elements in data blob does not match tensor length",
                        'AI.TENSORSET', 'z{0}', 'STRING', 2, 'BLOB', 'C-string\0followed by a non C-string')

    # ERR in bool tensor blob - element is not 0/1
    check_error_message(env, con, "BOOL tensor elements must be 0 or 1",
                        'AI.TENSORSET', 'z{0}', 'BOOL', 2, 'BLOB', "\x02\x01")


def test_common_tensorget(env):
    con = get_connection(env, '{0}')
    tested_datatypes = ["FLOAT", "DOUBLE", "INT8", "INT16", "INT32", "INT64", "UINT8", "UINT16", "BOOL"]
    tested_datatypes_fp = ["FLOAT", "DOUBLE"]
    tested_datatypes_int = ["INT8", "INT16", "INT32", "INT64", "UINT8", "UINT16", "BOOL"]
    for datatype in tested_datatypes:
        ret = con.execute_command('AI.TENSORSET', 'tensor_{}{{0}}'.format(datatype), datatype, 2, 'VALUES', 1, 1)
        env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    # AI.TENSORGET in BLOB format and set in a new key
    for datatype in tested_datatypes:
        tensor_blob = con.execute_command('AI.TENSORGET', 'tensor_{}{{0}}'.format(datatype), 'BLOB')
        ret = con.execute_command('AI.TENSORSET', 'tensor_blob_{}{{0}}'.format(datatype), datatype, 2, 'BLOB', tensor_blob)
        env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    reply_types = ["META", "VALUES", "BLOB"]
    # Confirm that tensor_{0} and tensor_blog_{0} are equal for META VALUES BLOB
    for datatype in tested_datatypes:
        for reply_type in reply_types:
            tensor_1_reply = con.execute_command('AI.TENSORGET', 'tensor_{}{{0}}'.format(datatype), reply_type)
            tensor_2_reply = con.execute_command('AI.TENSORGET', 'tensor_blob_{}{{0}}'.format(datatype), reply_type)
            env.assertEqual(tensor_1_reply, tensor_2_reply)

    # Confirm that the output is the expected for META
    for datatype in tested_datatypes:
        _, tensor_dtype, _, tensor_dim = con.execute_command('AI.TENSORGET', 'tensor_{}{{0}}'.format(datatype), "META")
        env.assertEqual(datatype.encode('utf-8'), tensor_dtype)
        env.assertEqual([2], tensor_dim)

    # Confirm that the output is the expected for VALUES
    for datatype in tested_datatypes:
        _, tensor_dtype, _, tensor_dim, _, tensor_values = con.execute_command('AI.TENSORGET', 'tensor_{}{{0}}'.format(datatype),
                                                                               'META', 'VALUES')
        env.assertEqual(datatype.encode('utf-8'), tensor_dtype)
        env.assertEqual([2], tensor_dim)
        if datatype in tested_datatypes_fp:
            env.assertEqual([b'1', b'1'], tensor_values)
        if datatype in tested_datatypes_int:
            env.assertEqual([1, 1], tensor_values)

    # Confirm that the output is the expected for BLOB
    for datatype in tested_datatypes:
        _, tensor_dtype, _, tensor_dim, _, tensor_blob = con.execute_command('AI.TENSORGET', 'tensor_{}{{0}}'.format(datatype),
                                                                             'META', 'BLOB')
        env.assertEqual(datatype.encode('utf-8'), tensor_dtype)
        env.assertEqual([2], tensor_dim)

    # Confirm that default reply format is META BLOB
    for datatype in tested_datatypes:
        _, tensor_dtype, _, tensor_dim, _, tensor_blob = con.execute_command('AI.TENSORGET', 'tensor_{}{{0}}'.format(datatype),
                                                                             'META', 'BLOB')
        _, tensor_dtype_default, _, tensor_dim_default, _, tensor_blob_default = con.execute_command('AI.TENSORGET',
                                                                                                     'tensor_{}{{0}}'.format(datatype))
        env.assertEqual(tensor_dtype, tensor_dtype_default)
        env.assertEqual(tensor_dim, tensor_dim_default)
        env.assertEqual(tensor_blob, tensor_blob_default)


def test_common_tensorget_error_replies(env):
    con = get_connection(env, '{0}')

    # ERR insufficient args
    check_error_message(env, con, "wrong number of arguments for 'AI.TENSORGET' command",
                        'AI.TENSORGET')

    # ERR too many arguments
    check_error_message(env, con, "wrong number of arguments for 'AI.TENSORGET' command",
                        'AI.TENSORGET', 'T_FLOAT{0}', 'META', 'VALUES', 'extra-arg')

    # ERR tensor key is empty
    check_error_message(env, con, "tensor key is empty or in a different shard",
                        'AI.TENSORGET', 'empty{0}')

    # WRONGTYPE Operation against a key holding the wrong kind of value
    con.execute_command('SET', 'non-tensor{0}', 'value')
    check_error_message(env, con, "WRONGTYPE Operation against a key holding the wrong kind of value",
                        'AI.TENSORGET', 'non-tensor{0}', 'unsupported')

    ret = con.execute_command('AI.TENSORSET', "T_FLOAT{0}", "FLOAT", 2, 'VALUES', 1, 1)
    env.assertEqual(ret, b'OK')

    # ERR unsupported data format
    check_error_message(env, con, "unsupported data format",
                        'AI.TENSORGET', 'T_FLOAT{0}', 'unsupported_format')

    # ERR invalid data format
    check_error_message(env, con, "both BLOB and VALUES specified",
                        'AI.TENSORGET', 'T_FLOAT{0}', 'BLOB', 'VALUES')


def test_common_tensorset_multiproc(env):
    run_test_multiproc(env, 'x', 10,
                       lambda env, i: env.execute_command('AI.TENSORSET', 'x', 'FLOAT', 2, 'VALUES', 2, 3))

    con = get_connection(env, 'x')
    ensureSlaveSynced(con, env)
    values = con.execute_command('AI.TENSORGET', 'x', 'VALUES')
    env.assertEqual(values, [b'2', b'3'])


def test_common_tensorset_multiproc_blob(env):
    con = get_connection(env, '{0}')
    tested_datatypes = ["FLOAT", "DOUBLE", "INT8", "INT16", "INT32", "INT64", "UINT8", "UINT16"]
    tested_datatypes_map = {}
    for datatype in tested_datatypes:
        ret = con.execute_command('AI.TENSORSET', 'tensor_{}{{0}}'.format(datatype), datatype, 1, 256)
        env.assertEqual(ret, b'OK')

    ensureSlaveSynced(con, env)

    # AI.TENSORGET in BLOB format and set in a new key
    for datatype in tested_datatypes:
        tensor_blob = con.execute_command('AI.TENSORGET', 'tensor_{}{{0}}'.format(datatype),
                                                                    'BLOB')
        ret = con.execute_command('AI.TENSORSET', 'tensor_blob_{}{{0}}'.format(datatype), datatype, 1, 256, 'BLOB', tensor_blob)
        tested_datatypes_map[datatype] = tensor_blob
        env.assertEqual(ret, b'OK')

    def funcname(env, blob, repetitions, same_key):
        for _ in range(1,same_key):
            for repetion in range(1,repetitions):
                env.execute_command('AI.TENSORSET', 'tensor_{}{{0}}'.format(repetitions), 'FLOAT', 1, 256, 'BLOB', blob)
    
    tensor_blob = tested_datatypes_map["FLOAT"]
    t = time.time()
    run_test_multiproc(env, '{0}', 10,
                       lambda env, i: funcname(env,tensor_blob,MAX_TRANSACTIONS,10) )
    elapsed_time = time.time() - t
    avg_ops_sec = 100000*10/elapsed_time
    # env.debugPrint("AI.TENSORSET elapsed time(sec) {:6.2f}\tAvg. ops/sec {:10.2f}".format(elapsed_time, avg_ops_sec), True)


def test_tensorset_disconnect(env):
    con = get_connection(env, 't_FLOAT')
    ret = send_and_disconnect(('AI.TENSORSET', 't_FLOAT', 'FLOAT', 2, 'VALUES', 2, 3), con)
    env.assertEqual(ret, None)


def test_tensorget_disconnect(env):
    con = get_connection(env, 't_FLOAT')
    ret = con.execute_command('AI.TENSORSET', 't_FLOAT', 'FLOAT', 2, 'VALUES', 2, 3)
    env.assertEqual(ret, b'OK')
    ret = send_and_disconnect(('AI.TENSORGET', 't_FLOAT', 'META'), con)
    env.assertEqual(ret, None)


def test_lua_multi(env):
    con = get_connection(env, '{1}')
    ret = con.execute_command('MULTI')
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('AI.MODELEXECUTE', "no_model{1}", "INPUTS", 1, "no_input{1}", "OUTPUTS", 1, "no_output{1}")
    env.assertEqual(ret, b'QUEUED')
    try:
        ret = con.execute_command('EXEC')
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("ERR Cannot run RedisAI command within a transaction or a LUA script", exception.__str__())
    try:
        ret = con.execute_command('EVAL', "return redis.pcall('AI.MODELEXECUTE', 'no_model{1}', 'INPUTS', 1, 'NO_INPUT{1}',"
                                          " 'OUTPUTS', 1, 'NO_OUTPUT{1}')", 0)
    except Exception as e:
        exception = e
        env.assertEqual(type(exception), redis.exceptions.ResponseError)
        env.assertEqual("Cannot run RedisAI command within a transaction or a LUA script", exception.__str__())


def test_info_command(env):
    con = get_connection(env, '{0}')
    versions = get_info_section(con, 'versions')
    env.assertEqual(list(versions.keys()), ['ai_RedisAI_version', 'ai_low_level_API_version', 'ai_rdb_version'])
    git = get_info_section(con, 'git')
    env.assertEqual(list(git.keys()), ['ai_git_sha'])
    load_time_configs = get_info_section(con, 'load_time_configs')
    env.assertEqual(list(load_time_configs.keys()), ['ai_threads_per_queue', 'ai_inter_op_parallelism',
                                                     'ai_intra_op_parallelism', 'ai_model_execution_timeout',
                                                     'ai_backend_memory_limit'])
    # minimum cpu properties
    cpu = get_info_section(con, 'cpu')
    env.assertTrue('ai_self_used_cpu_sys' in cpu.keys())
    env.assertTrue('ai_self_used_cpu_user' in cpu.keys())
    env.assertTrue('ai_children_used_cpu_sys' in cpu.keys())
    env.assertTrue('ai_children_used_cpu_user' in cpu.keys())
    env.assertTrue('ai_queue_CPU_bthread_n1_used_cpu_total' in cpu.keys())


def test_string_tensor(env):
    con = get_connection(env, '{0}')

    # test creation of string tensor from values - every C-string is valid (null-terminated)
    # we also allow omitting the final null-character from the string
    ret = con.execute_command('AI.TENSORSET', 'string_tensor_from_val{0}', 'STRING', 2, 'VALUES', 'str_val1', 'str_val2\0')
    env.assertEqual(ret, b'OK')

    tensor_reply_values = con.execute_command('AI.TENSORGET', 'string_tensor_from_val{0}', 'VALUES')
    tensor_reply_blob = con.execute_command('AI.TENSORGET', 'string_tensor_from_val{0}', 'BLOB')
    env.assertEqual(tensor_reply_values, [b'str_val1', b'str_val2'])
    env.assertEqual(tensor_reply_blob, b'str_val1\0str_val2\0')

    # test creation of string tensor from blob
    ret = con.execute_command('AI.TENSORSET', 'string_tensor_from_blob{0}', 'STRING', 2, 'BLOB', 'str_blob1\0str_blob2\0')
    env.assertEqual(ret, b'OK')

    tensor_reply_values = con.execute_command('AI.TENSORGET', 'string_tensor_from_blob{0}', 'VALUES')
    tensor_reply_blob = con.execute_command('AI.TENSORGET', 'string_tensor_from_blob{0}', 'BLOB')
    env.assertEqual(tensor_reply_values, [b'str_blob1', b'str_blob2'])
    env.assertEqual(tensor_reply_blob, b'str_blob1\0str_blob2\0')

    # test for empty string tensor
    ret = con.execute_command('AI.TENSORSET', 'empty_string_tensor{0}', 'STRING', 2)
    env.assertEqual(ret, b'OK')
    _, tensor_dtype, _, tensor_dim, _, tensor_values = con.execute_command('AI.TENSORGET', 'empty_string_tensor{0}', 'META', 'VALUES')
    env.assertEqual(tensor_dtype, b'STRING')
    env.assertEqual(tensor_dim, [2])
    env.assertEqual(tensor_values, [b'', b''])

    # advanced - tensor with more than one dimension
    ret = con.execute_command('AI.TENSORSET', 'string_tensor_few_dims{0}', 'STRING', 2, 2, 'BLOB',
                              'str11\0str12\0str21\0str22\0')
    env.assertEqual(ret, b'OK')

    _, tensor_dtype, _, tensor_dim, _, tensor_values = con.execute_command('AI.TENSORGET', 'string_tensor_few_dims{0}', 'META', 'VALUES')
    env.assertEqual(tensor_dtype, b'STRING')
    env.assertEqual(tensor_dim, [2, 2])
    env.assertEqual(tensor_values, [b'str11', b'str12', b'str21', b'str22'])

    # advanced - tensor with non ascii characters
    ret = con.execute_command('AI.TENSORSET', 'string_tensor_non-ascii{0}', 'STRING', 2, 'BLOB',
                              'english\0עברית\0')
    env.assertEqual(ret, b'OK')

    tensor_reply_values = con.execute_command('AI.TENSORGET', 'string_tensor_non-ascii{0}', 'VALUES')
    env.assertEqual(tensor_reply_values[1].decode('utf-8'), 'עברית')
