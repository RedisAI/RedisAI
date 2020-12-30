
def redis_string_int_to_tensor(redis_value: Any):
    return torch.tensor(int(str(redis_value)))


def redis_string_float_to_tensor(redis_value: Any):
    return torch.tensor(float(str((redis_value))))


# def redis_int_to_tensor(redis_value: RedisValue):
#     return tensor(redis_value.intValue())


# def redis_int_list_to_tensor(redis_value: RedisValue):
#     len = len(redis_value.getList())
#     l = []
#     for v in redis_value.getList():
#         l.append(redis_string_to_int(v))
#     return torch.cat(l, dim=0)


# def redis_float_list_to_tensor(redis_value: RedisValue):
#     len = len(redis_value.getList())
#     l = []
#     for v in redis_value.getList():
#         l.append(redis_string_to_float(v))
#     return torch.cat(l, dim=0)


# def redis_hash_to_tensor(redis_value: RedisValue):
#     len = len(redis_value.getList())
#     l = []
#     for v in redis_value.getList():
#         l.append(redis_string_to_float(v.getList()[1]))
#     return torch.cat(l, dim=0)

# def test_redis_error():
#     res = redis.executeCommand("SET", "x")
#     return tensor(res.getValueType())

def test_int_set_get():
    redis.execute("SET", "x", "1")
    res = redis.execute("GET", "x",)
    redis.execute("DEL", "x")
    return redis_string_int_to_tensor(res)

def test_float_set_get():
    redis.execute("SET", "x", "1.1")
    res = redis.execute("GET", "x",)
    redis.execute("DEL", "x")
    return redis_string_float_to_tensor(res)

# def test_int_list():
#     redis.executeCommand("LPUSH", "x", "1")
#     redis.executeCommand("LPUSH", "x", "2")
#     res = redis.executeCommand("LRANGE", "x")
#     redis.executeCommand("DEL", "x")
#     return redis_int_list_to_tensor(res)

# def test_float_list():
#     redis.executeCommand("LPUSH", "x", "1.1")
#     redis.executeCommand("LPUSH", "x", "2.2")
#     res = redis.executeCommand("LRANGE", "x")
#     redis.executeCommand("DEL", "x")
#     return redis_float_list_to_tensor(res)

# def test_hash():
#     redis.executeCommand("HSET", "x", "1", "2.2)
#     res = redis.executeCommand("HGETALL", "x")
#     redis.executeCommand("DEL", "x")
#     return redis_float_list_to_tensor(res)


def test_set_key():
    redis.execute("SET", ["x", "1"])


def test_del_key():
    redis.execute("DEL", ["x"])
