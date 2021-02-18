#include "tensor_type.h"
#include "tensor.h"
#include "../serialization/AOF/rai_aof_rewrite.h"
#include "../serialization/RDB/encoder/rai_rdb_encode.h"
#include "../serialization/RDB/decoder/rai_rdb_decoder.h"
#include "../serialization/RDB/decoder/decode_previous.h"

extern RedisModuleType *RedisAI_TensorType;

static void RAI_Tensor_RdbSave(RedisModuleIO *io, void *value) { RAI_RDBSaveTensor(io, value); }

static void *RAI_Tensor_RdbLoad(struct RedisModuleIO *io, int encver) {
    if (encver > REDISAI_ENC_VER) {
        RedisModule_LogIOError(
            io, "error", "Failed loading tensor, RedisAI version (%d) is not forward compatible.\n",
            REDISAI_MODULE_VERSION);
        return NULL;
    } else if (encver < REDISAI_ENC_VER) {
        return Decode_PreviousTensor(io, encver);
    } else {
        return RAI_RDBLoadTensor(io);
    }
}

static void RAI_Tensor_AofRewrite(RedisModuleIO *aof, RedisModuleString *key, void *value) {
    RAI_AOFRewriteTensor(aof, key, value);
}

static void RAI_Tensor_DTFree(void *value) { RAI_TensorFree(value); }

int TensorType_Register(RedisModuleCtx *ctx) {
    RedisModuleTypeMethods tmTensor = {
        .version = REDISMODULE_TYPE_METHOD_VERSION,
        .rdb_load = RAI_Tensor_RdbLoad,
        .rdb_save = RAI_Tensor_RdbSave,
        .aof_rewrite = RAI_Tensor_AofRewrite,
        .mem_usage = NULL,
        .free = RAI_Tensor_DTFree,
        .digest = NULL,
    };
    RedisAI_TensorType = RedisModule_CreateDataType(ctx, "AI_TENSOR", REDISAI_ENC_VER, &tmTensor);
    return RedisAI_TensorType != NULL;
}
