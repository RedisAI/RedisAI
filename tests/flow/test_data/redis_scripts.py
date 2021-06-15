
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

def test_redis_error(key:str):
    redis.execute("SET", key)

def test_int_set_get(key:str, value:int):
    redis.execute("SET", key, str(value))
    res = redis.execute("GET", key)
    redis.execute("DEL", key)
    return redis_string_int_to_tensor(res)

def test_int_set_incr(key:str, value:int):
    redis.execute("SET", key, str(value))
    res = redis.execute("INCR", key)
    redis.execute("DEL", key)
    return redis_string_int_to_tensor(res)

def test_float_set_get(key:str, value:float):
    redis.execute("SET", key, str(value))
    res = redis.execute("GET", key)
    redis.execute("DEL", key)
    return redis_string_float_to_tensor(res)

def test_int_list(key:str, l:List[str]):
    for value in l:
        redis.execute("RPUSH", key, value)
    res = redis.execute("LRANGE", key, "0", str(len(l)))
    redis.execute("DEL", key)
    return redis_int_list_to_tensor(res)


def test_str_list(key:str, l:List[str]):
    for value in l:
        redis.execute("RPUSH", key, value)


def test_hash(key:str, l:List[str]):
    args = [key]
    for s in l:
        args.append(s)
    redis.execute("HSET", args)
    res = redis.execute("HVALS", key)
    redis.execute("DEL", key)
    return redis_hash_to_tensor(res)


def test_set_key(key:str, value:str):
    redis.execute("SET", [key, value])


def test_del_key(key:str):
    redis.execute("DEL", [key])


def test_model_execute(keys:List[str]):
    a = torch.tensor([[2.0, 3.0], [2.0, 3.0]])
    b = torch.tensor([[2.0, 3.0], [2.0, 3.0]])
    return redisAI.model_execute(keys[0], [a, b], 1)  # assume keys[0] is the model key name saved in redis
