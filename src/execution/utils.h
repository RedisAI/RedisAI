#pragma once
#include "redismodule.h"
#include "redis_ai_objects/tensor_struct.h"
#include "redis_ai_objects/model_struct.h"
#include "redis_ai_objects/err.h"
#include <stdbool.h>

/** Use this to check if a command is given a key whose hash slot is not on the current
 *  shard, when using enterprise cluster.
 **/
bool VerifyKeyInThisShard(RedisModuleCtx *ctx, RedisModuleString *key_str);

/**
 * Use this function when loading the model. Stores the version in global variables.
 */
void getRedisVersion();

/**
 * Returns true if Redis is running in enterprise mode.
 */
int IsEnterprise();