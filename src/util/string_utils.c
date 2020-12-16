#include "string_utils.h"

RedisModuleString *RAI_HoldString(RedisModuleCtx *ctx, RedisModuleString *str) {
    if (str == NULL) {
        return NULL;
    }
    RedisModuleString *out;
    if (RMAPI_FUNC_SUPPORTED(RedisModule_HoldString)) {
        out = RedisModule_HoldString(NULL, str);
    } else {
        RedisModule_RetainString(NULL, str);
        out = str;
    }
    return out;
}
