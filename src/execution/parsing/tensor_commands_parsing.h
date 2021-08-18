#pragma once

#include "redismodule.h"
#include "redis_ai_objects/err.h"
#include "redis_ai_objects/tensor.h"

/**
 * Helper method to parse AI.TENSORGET arguments
 *
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @param t Destination tensor to store the parsed data
 * @param error error data structure to store error message in the case of
 * parsing failures
 * @return REDISMODULE_OK on success, or REDISMODULE_ERR if the parsing failed
 */
int ParseTensorSetArgs(RedisModuleString **argv, int argc, RAI_Tensor **t, RAI_Error *error);

/**
 * Helper method to parse AI.TENSORGET arguments
 *
 * @param error error data structure to store error message in the case of
 * parsing failures
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @return The format in which tensor is returned.
 */

uint ParseTensorGetArgs(RAI_Error *error, RedisModuleString **argv, int argc);
