
# def redis_string_int_to_tensor(redis_value: Any):
#     return torch.tensor(int(str(redis_value)))


# def redis_string_float_to_tensor(redis_value: Any):
#     return torch.tensor(float(str((redis_value))))


# def redis_int_to_tensor(redis_value: int):
#     return torch.tensor(redis_value)


def redis_int_list_to_tensor(redis_value: Any):
    values = redis.asList(redis_value)
    l  = [torch.tensor(int(str(v))).reshape(1,1) for v in values]
    return torch.cat(l, dim=0)


# def redis_hash_to_tensor(redis_value: Any):
#     values = redis.asList(redis_value)
#     l  = [torch.tensor(int(str(v))).reshape(1,1) for v in values]
#     return torch.cat(l, dim=0)

# def test_redis_error(key:str):
#     redis.execute("SET", key)

# def test_int_set_get(key:str, value:str):
#     redis.execute("SET", key, value)
#     res = redis.execute("GET", key,)
#     redis.execute("DEL", key)
#     return redis_string_int_to_tensor(res)

# def test_int_set_incr(key:str, value:str):
#     redis.execute("SET", key, value)
#     res = redis.execute("INCR", key)
#     redis.execute("DEL", key)
#     return redis_string_int_to_tensor(res)

# def test_float_set_get(key:str, value:str):
#     redis.execute("SET", key, value)
#     res = redis.execute("GET", key,)
#     redis.execute("DEL", key)
#     return redis_string_float_to_tensor(res)

# def test_int_list(key:str):
#     redis.execute("RPUSH", key, "1")
#     redis.execute("RPUSH", key, "2")
#     res = redis.execute("LRANGE", key, "0", "2")
#     redis.execute("DEL", key)
#     return redis_int_list_to_tensor(res)


def test_str_list(key:str, l:List[str]):
    redis.execute("RPUSH", key, "1")
    redis.execute("RPUSH", key, "2")
    res = redis.execute("LRANGE", key, "0", "2")
    redis.execute("DEL", key)
    return redis_int_list_to_tensor(res)

# def test_hash(key:str):
#     redis.execute("HSET", key, "field1", "1", "field2", "2")
#     res = redis.execute("HVALS", key)
#     redis.execute("DEL", key)
#     return redis_hash_to_tensor(res)


# def test_set_key(key:str, value:str):
#     redis.execute("SET", [key, value])


# def test_del_key(key:str):
#     redis.execute("DEL", [key])
