#pragma once
#include "redismodule.h"
#include "redis_ai_objects/tensor_struct.h"
#include "redis_ai_objects/model_struct.h"
#include "redis_ai_objects/err.h"
#include <stdbool.h>

/* Use this to check if a command is given a key whose hash slot is not on the current
   shard, when using enterprise cluster.*/
bool VerifyKeyInThisShard(RedisModuleCtx *ctx, RedisModuleString *key_str);

/**
 * Helper method to get Tensor from keyspace. In case of a failure an
 * error is documented.
 *
 * @param ctx Context in which Redis modules operate
 * @param keyName key name
 * @param key tensor's key handle. On success it contains an handle representing
 * a Redis key with the requested access mode
 * @param tensor destination tensor structure
 * @param mode key access mode
 * @return REDISMODULE_OK if the tensor value stored at key was correctly
 * returned and available at *tensor variable, or REDISMODULE_ERR if there was
 * an error getting the Tensor
 */
int RAI_GetTensorFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName, RedisModuleKey **key,
                              RAI_Tensor **tensor, int mode, RAI_Error *err);

/**
 * Helper method to get a Model from keyspace. In the case of failure the key is
 * closed and the error is replied ( no cleaning actions required )
 *
 * @param ctx Context in which Redis modules operate
 * @param keyName key name
 * @param model destination model structure
 * @param mode key access mode
 * @param error contains the error in case of problem with retrival
 * @return REDISMODULE_OK if the model value stored at key was correctly
 * returned and available at *model variable, or REDISMODULE_ERR if there was
 * an error getting the Model
 */
int RAI_GetModelFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName, RAI_Model **model,
                             int mode, RAI_Error *err);
