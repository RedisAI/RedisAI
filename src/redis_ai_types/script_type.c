/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "script_type.h"
#include "redis_ai_objects/script.h"
#include "serialization/AOF/rai_aof_rewrite.h"
#include "serialization/RDB/encoder/rai_rdb_encode.h"
#include "serialization/RDB/decoder/rai_rdb_decoder.h"
#include "serialization/RDB/decoder/decode_previous.h"

extern RedisModuleType *RedisAI_ScriptType;

static void *RAI_Script_RdbLoad(struct RedisModuleIO *io, int encver) {
    if (encver > REDISAI_ENC_VER) {
        RedisModule_LogIOError(
            io, "error", "Failed loading script, RedisAI version (%d) is not forward compatible.\n",
            REDISAI_MODULE_VERSION);
        return NULL;
    } else if (encver < REDISAI_ENC_VER) {
        return Decode_PreviousScript(io, encver);
    } else {
        return RAI_RDBLoadScript(io);
    }
}

static void RAI_Script_RdbSave(RedisModuleIO *io, void *value) { RAI_RDBSaveScript(io, value); }

static void RAI_Script_AofRewrite(RedisModuleIO *aof, RedisModuleString *key, void *value) {
    RAI_AOFRewriteScript(aof, key, value);
}

static void RAI_Script_DTFree(void *value) {
    RAI_Error err = {0};
    RAI_ScriptFree(value, &err);
    if (err.code != RAI_OK) {
        printf("ERR: %s\n", err.detail);
        RAI_ClearError(&err);
    }
}

int ScriptType_Register(RedisModuleCtx *ctx) {
    RedisModuleTypeMethods tmScript = {.version = REDISMODULE_TYPE_METHOD_VERSION,
                                       .rdb_load = RAI_Script_RdbLoad,
                                       .rdb_save = RAI_Script_RdbSave,
                                       .aof_rewrite = RAI_Script_AofRewrite,
                                       .mem_usage = NULL,
                                       .free = RAI_Script_DTFree,
                                       .digest = NULL};

    RedisAI_ScriptType = RedisModule_CreateDataType(ctx, "AI_SCRIPT", REDISAI_ENC_VER, &tmScript);
    return RedisAI_ScriptType != NULL;
}
