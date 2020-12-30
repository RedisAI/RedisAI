
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

def test_redis_error():
    redis.execute("SET", "x")

def test_int_set_get():
    redis.execute("SET", "x", "1")
    res = redis.execute("GET", "x",)
    redis.execute("DEL", "x")
    return redis_string_int_to_tensor(res)

def test_int_set_incr():
    redis.execute("SET", "x", "1")
    res = redis.execute("INCR", "x")
    redis.execute("DEL", "x")
    return redis_string_int_to_tensor(res)

def test_float_set_get():
    redis.execute("SET", "x", "1.1")
    res = redis.execute("GET", "x",)
    redis.execute("DEL", "x")
    return redis_string_float_to_tensor(res)

def test_int_list():
    redis.execute("RPUSH", "x", "1")
    redis.execute("RPUSH", "x", "2")
    res = redis.execute("LRANGE", "x", "0", "2")
    redis.execute("DEL", "x")
    return redis_int_list_to_tensor(res)


def test_hash():
    redis.execute("HSET", "x", "field1", "1", "field2", "2")
    res = redis.execute("HVALS", "x")
    redis.execute("DEL", "x")
    return redis_hash_to_tensor(res)


def test_set_key():
    redis.execute("SET", ["x{1}", "1"])


def test_del_key():
    redis.execute("DEL", ["x"])
