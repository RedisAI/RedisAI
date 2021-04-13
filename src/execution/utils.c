#include "utils.h"
#include "redis_ai_objects/tensor.h"
#include "redis_ai_objects/model.h"

extern int rlecMajorVersion;

static inline int IsEnterprise() { return rlecMajorVersion != -1; }

bool VerifyKeyInThisShard(RedisModuleCtx *ctx, RedisModuleString *key_str) {

    if (IsEnterprise()) {
        int first_slot, last_slot;
        RedisModule_ShardingGetSlotRange(&first_slot, &last_slot);
        int key_slot = RedisModule_ShardingGetKeySlot(key_str);
        if (key_slot < first_slot || key_slot > last_slot) {
            RedisModule_Log(ctx, "warning",
                            "could not load %s from keyspace,"
                            " this key's hash slot belongs to a different shard",
                            RedisModule_StringPtrLen(key_str, NULL));
            return false;
        }
    }
    RedisModule_Log(ctx, "warning", "could not load %s from keyspace, key doesn't exist",
                    RedisModule_StringPtrLen(key_str, NULL));
    return true;
}

int RAI_GetTensorFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName, RedisModuleKey **key,
                              RAI_Tensor **tensor, int mode, RAI_Error *err) {
    *key = RedisModule_OpenKey(ctx, keyName, mode);
    if (RedisModule_KeyType(*key) == REDISMODULE_KEYTYPE_EMPTY) {
        RedisModule_CloseKey(*key);
        if (VerifyKeyInThisShard(ctx, keyName)) { // Relevant for enterprise cluster.
            RAI_SetError(err, RAI_EKEYEMPTY, "ERR tensor key is empty");
        } else {
            RAI_SetError(err, RAI_EKEYEMPTY,
                         "ERR CROSSSLOT Keys in request don't hash to the same slot");
        }
        return REDISMODULE_ERR;
    }
    if (RedisModule_ModuleTypeGetType(*key) != RedisAI_TensorType) {
        RedisModule_CloseKey(*key);
        RedisModule_Log(ctx, "error", "%s is not a tensor",
                        RedisModule_StringPtrLen(keyName, NULL));
        RAI_SetError(err, RAI_ETENSORGET, REDISMODULE_ERRORMSG_WRONGTYPE);
        return REDISMODULE_ERR;
    }
    *tensor = RedisModule_ModuleTypeGetValue(*key);
    RedisModule_CloseKey(*key);
    return REDISMODULE_OK;
}

int RAI_GetModelFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName, RAI_Model **model,
                             int mode, RAI_Error *err) {
    RedisModuleKey *key = RedisModule_OpenKey(ctx, keyName, mode);
    if (RedisModule_KeyType(key) == REDISMODULE_KEYTYPE_EMPTY) {
        RedisModule_CloseKey(key);
        // #IFDEF LITE
        if (VerifyKeyInThisShard(ctx, keyName)) { // Relevant for enterprise cluster.
            RAI_SetError(err, RAI_EKEYEMPTY, "ERR model key is empty");
        } else {
            RAI_SetError(err, RAI_EKEYEMPTY,
                         "ERR CROSSSLOT Keys in request don't hash to the same slot");
        }
        // #ELSE
        RedisModule_Log(ctx, "error", "could not load %s from keyspace, key doesn't exist",
                        RedisModule_StringPtrLen(keyName, NULL));
        RAI_SetError(err, RAI_EKEYEMPTY, "ERR model key is empty");
        // #ENDIF
        return REDISMODULE_ERR;
    }
    if (RedisModule_ModuleTypeGetType(key) != RedisAI_ModelType) {
        RedisModule_CloseKey(key);
        RedisModule_Log(ctx, "error", "%s is not a model", RedisModule_StringPtrLen(keyName, NULL));
        RAI_SetError(err, RAI_EMODELRUN, REDISMODULE_ERRORMSG_WRONGTYPE);
        return REDISMODULE_ERR;
    }
    *model = RedisModule_ModuleTypeGetValue(key);
    RedisModule_CloseKey(key);
    return REDISMODULE_OK;
}
