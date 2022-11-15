/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "redismodule.h"
#include "redis_ai_objects/err.h"

/**
 * @brief  Parse and validate TIMEOUT argument. If it is valid, store it in timeout.
 * Otherwise set an error.
 * @return Returns REDISMODULE_OK if the command is valid, REDISMODULE_ERR otherwise.
 */
int ParseTimeout(RedisModuleString *timeout_arg, RAI_Error *error, long long *timeout);

/**
 * @brief
 *
 * @param functionName
 * @return const char*
 */
const char *ScriptCommand_GetFunctionName(RedisModuleString *functionName);

/**
 * Parse KEYS section in command [* KEYS <nkeys> key1 key2... ]
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @param err An error object to store an error message if needed.
 * @return processed number of arguments on success, or -1 if the parsing failed
 */

int ValidateKeysArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc, RAI_Error *err);
