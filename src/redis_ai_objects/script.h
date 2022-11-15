/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

/**
 * script.h
 *
 * Contains headers for the helper methods for both creating, populating,
 * managing and destructing the PyTorch Script data structure.
 *
 */

#pragma once

#include "err.h"
#include "tensor.h"
#include "script_struct.h"
#include "redismodule.h"

/**
 * Helper method to allocated and initialize a RAI_Script. Relies on Pytorch
 * backend `script_create` callback function.
 *
 * @param devicestr device string
 * @param tag script model tag
 * @param scriptdef encoded script definition
 * @param err error data structure to store error message in the case of
 * failures
 * @return RAI_Script script structure on success, or NULL if failed
 */
RAI_Script *RAI_ScriptCreate(const char *devicestr, RedisModuleString *tag, const char *scriptdef,
                             RAI_Error *err);

/**
 * @brief Helper method to allocated and initialize a RAI_Script with entrypoints. Relies on Pytorch
 * backend `script_create` callback function.
 *
 * @param devicestr device string
 * @param tag script model tag
 * @param scriptdef encoded script definition
 * @param entryPoints array of entry point function names
 * @param nEntryPoints number of entry points
 * @param err error data structure to store error message in the case of
 * failures
 * @return RAI_Script*
 */
RAI_Script *RAI_ScriptCompile(const char *devicestr, RedisModuleString *tag, const char *scriptdef,
                              const char **entryPoints, size_t nEntryPoints, RAI_Error *err);

/**
 * Frees the memory of the RAI_Script when the script reference count reaches
 * 0. It is safe to call this function with a NULL input script.
 *
 * @param script input script to be freed
 * @param error error data structure to store error message in the case of
 * failures
 */
void RAI_ScriptFree(RAI_Script *script, RAI_Error *err);

/**
 * Every call to this function, will make the RAI_Script 'script' requiring an
 * additional call to RAI_ScriptFree() in order to really free the script.
 * Returns a shallow copy of the script.
 *
 * @param script input script
 * @return script
 */
RAI_Script *RAI_ScriptGetShallowCopy(RAI_Script *script);

/* Return REDISMODULE_ERR if there was an error getting the Script.
 * Return REDISMODULE_OK if the model value stored at key was correctly
 * returned and available at *model variable. */

/**
 * Helper method to get a Script from keyspace. In the case of failure the key
 * is closed and the error is replied ( no cleaning actions required )
 *
 * @param ctx Context in which Redis modules operate
 * @param keyName key name
 * @param script destination script structure
 * @param mode key access mode
 * @return REDISMODULE_OK if the script value stored at key was correctly
 * returned and available at *script variable, or REDISMODULE_ERR if there was
 * an error getting the Script
 */
int RAI_GetScriptFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName, RAI_Script **script,
                              int mode, RAI_Error *err);

/**
 * When a module command is called in order to obtain the position of
 * keys, since it was flagged as "getkeys-api" during the registration,
 * the command implementation checks for this special call using the
 * RedisModule_IsKeysPositionRequest() API and uses this function in
 * order to report keys.
 * No real execution is done on this special call.
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @return
 */
int RedisAI_ScriptRun_IsKeysPositionRequest_ReportKeys(RedisModuleCtx *ctx,
                                                       RedisModuleString **argv, int argc);

/**
 * When a module command is called in order to obtain the position of
 * keys, since it was flagged as "getkeys-api" during the registration,
 * the command implementation checks for this special call using the
 * RedisModule_IsKeysPositionRequest() API and uses this function in
 * order to report keys.
 * No real execution is done on this special call.
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @return
 */
int RedisAI_ScriptExecute_IsKeysPositionRequest_ReportKeys(RedisModuleCtx *ctx,
                                                           RedisModuleString **argv, int argc);

/**
 * @brief  Returns the redis module type representing a script.
 * @return redis module type representing a script.
 */
RedisModuleType *RAI_ScriptRedisType(void);
