#include "model_type.h"
#include "redis_ai_objects/model.h"
#include "serialization/AOF/rai_aof_rewrite.h"
#include "serialization/RDB/encoder/rai_rdb_encode.h"
#include "serialization/RDB/decoder/rai_rdb_decoder.h"
#include "serialization/RDB/decoder/decode_previous.h"

extern RedisModuleType *RedisAI_ModelType;

static void *RAI_Model_RdbLoad(struct RedisModuleIO *io, int encver) {
    if (encver > REDISAI_ENC_VER) {
        RedisModule_LogIOError(
            io, "error", "Failed loading model, RedisAI version (%d) is not forward compatible.\n",
            REDISAI_MODULE_VERSION);
        return NULL;
    } else if (encver < REDISAI_ENC_VER) {
        return Decode_PreviousModel(io, encver);
    } else {
        return RAI_RDBLoadModel(io);
    }
}

static void RAI_Model_RdbSave(RedisModuleIO *io, void *value) { RAI_RDBSaveModel(io, value); }

static void RAI_Model_AofRewrite(RedisModuleIO *aof, RedisModuleString *key, void *value) {
    RAI_AOFRewriteModel(aof, key, value);
}

static void RAI_Model_DTFree(void *value) {
    RAI_Error err = {0};
    RAI_Model *model = (RAI_Model *)value;
    RAI_RemoveStatsEntry(model->infokey);
    RAI_ModelFree(model, &err);
    if (err.code != RAI_OK) {
        printf("ERR: %s\n", err.detail);
        RAI_ClearError(&err);
    }
}

int ModelType_Register(RedisModuleCtx *ctx) {
    RedisModuleTypeMethods tmModel = {.version = REDISMODULE_TYPE_METHOD_VERSION,
                                      .rdb_load = RAI_Model_RdbLoad,
                                      .rdb_save = RAI_Model_RdbSave,
                                      .aof_rewrite = RAI_Model_AofRewrite,
                                      .mem_usage = NULL,
                                      .free = RAI_Model_DTFree,
                                      .digest = NULL};

    RedisAI_ModelType = RedisModule_CreateDataType(ctx, "AI__MODEL", REDISAI_ENC_VER, &tmModel);
    return RedisAI_ModelType != NULL;
}
