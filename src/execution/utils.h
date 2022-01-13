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
void RedisAI_SetRedisVersion();

/**
 * Returns redis version in the major, minor, and patch placeholders.
 */
void RedisAI_GetRedisVersion(int *major, int *minor, int *patch);

/**
 * Returns true if Redis is running in enterprise mode.
 */
bool IsEnterprise();
