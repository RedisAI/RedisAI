
def redis_string_int_to_tensor(redis_value: Any):
    return torch.tensor(int(str(redis_value)))


def redis_string_float_to_tensor(redis_value: Any):
    return torch.tensor(float(str((redis_value))))


def redis_int_to_tensor(redis_value: int):
    return torch.tensor(redis_value)


def redis_int_list_to_tensor(redis_value: Any):
    values = redis.asList(redis_value)
    l  = [torch.tensor(int(str(v))).reshape(1,1) for v in values]
    return torch.cat(l, dim=0)


def redis_hash_to_tensor(redis_value: Any):
    values = redis.asList(redis_value)
    l  = [torch.tensor(int(str(v))).reshape(1,1) for v in values]
    return torch.cat(l, dim=0)

def test_redis_error(tensors: List[Tensor], keys: List[str], args: List[str]):
    key = keys[0]
    redis.execute("SET", key)

def test_int_set_get(tensors: List[Tensor], keys: List[str], args: List[str]):
    key = keys[0]
    value = args[0]
    redis.execute("SET", key, value)
    res = redis.execute("GET", key)
    redis.execute("DEL", key)
    return redis_string_int_to_tensor(res)

def test_int_set_incr(tensors: List[Tensor], keys: List[str], args: List[str]):
    key = keys[0]
    value = args[0]
    redis.execute("SET", key, value)
    res = redis.execute("INCR", key)
    redis.execute("DEL", key)
    return redis_string_int_to_tensor(res)

def test_float_set_get(tensors: List[Tensor], keys: List[str], args: List[str]):
    key = keys[0]
    value = args[0]
    redis.execute("SET", key, value)
    res = redis.execute("GET", key)
    redis.execute("DEL", key)
    return redis_string_float_to_tensor(res)

def test_int_list(tensors: List[Tensor], keys: List[str], args: List[str]):
    key = keys[0]
    for value in args:
        redis.execute("RPUSH", key, value)
    res = redis.execute("LRANGE", key, "0", str(len(args)))
    redis.execute("DEL", key)
    return redis_int_list_to_tensor(res)


def test_str_list(tensors: List[Tensor], keys: List[str], args: List[str]):
    key = keys[0]
    for value in args:
        redis.execute("RPUSH", key, value)


def test_hash(tensors: List[Tensor], keys: List[str], args: List[str]):
    key = keys[0]
    command_args = [key]
    for s in args:
        command_args.append(s)
    redis.execute("HSET", command_args)
    res = redis.execute("HVALS", key)
    redis.execute("DEL", key)
    return redis_hash_to_tensor(res)


def test_set_key(tensors: List[Tensor], keys: List[str], args: List[str]):
    key = keys[0]
    value = args[0]
    redis.execute("SET", [key, value])


def test_del_key(tensors: List[Tensor], keys: List[str], args: List[str]):
    key = keys[0]
    redis.execute("DEL", [key])


def test_model_execute(tensors: List[Tensor], keys: List[str], args: List[str]):
    a = torch.tensor([[2.0, 3.0], [2.0, 3.0]])
    b = torch.tensor([[2.0, 3.0], [2.0, 3.0]])
    return redisAI.model_execute(keys[0], [a, b], 1)  # assume keys[0] is the model key name saved in redis


def test_model_execute_onnx(tensors: List[Tensor], keys: List[str], args: List[str]):
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    return redisAI.model_execute(keys[0], [a], 1)  # assume keys[0] is the model key name saved in redis


def test_model_execute_onnx_bad_input(tensors: List[Tensor], keys: List[str], args: List[str]):
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return redisAI.model_execute(keys[0], [a], 1)  # assume keys[0] is the model key name saved in redis
