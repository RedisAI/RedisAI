#pragma once

#include <stdbool.h>
#include "redismodule.h"

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
bool IsEnterprise();
