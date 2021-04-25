#pragma once
#include "redismodule.h"
#include "redis_ai_objects/err.h"

/**
 * @brief  Parse and validate TIMEOUT argument. If it is valid, store it in timeout.
 * Otherwise set an error.
 * @return Returns REDISMODULE_OK if the command is valid, REDISMODULE_ERR otherwise.
 */
int ParseTimeout(RedisModuleString *timeout_arg, RAI_Error *error, long long *timeout);
