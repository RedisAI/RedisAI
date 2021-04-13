#include "string_utils.h"
#include "dict.h"
#include <string.h>
#include "util/redisai_memory.h"

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

uint64_t RAI_StringsHashFunction(const void *key) {
    return AI_dictGenHashFunction(key, strlen((char *)key));
}

int RAI_StringsKeyCompare(void *privdata, const void *key1, const void *key2) {
    const char *strKey1 = key1;
    const char *strKey2 = key2;

    return strcmp(strKey1, strKey2) == 0;
}

void RAI_StringsKeyDestructor(void *privdata, void *key) { RA_FREE(key); }

void *RAI_StringsKeyDup(void *privdata, const void *key) { return RA_STRDUP((char *)key); }

uint64_t RAI_RStringsHashFunction(const void *key) {
    size_t len;
    const char *buffer = RedisModule_StringPtrLen((RedisModuleString *)key, &len);
    return AI_dictGenHashFunction(buffer, len);
}

int RAI_RStringsKeyCompare(void *privdata, const void *key1, const void *key2) {
    RedisModuleString *strKey1 = (RedisModuleString *)key1;
    RedisModuleString *strKey2 = (RedisModuleString *)key2;

    return RedisModule_StringCompare(strKey1, strKey2) == 0;
}

void RAI_RStringsKeyDestructor(void *privdata, void *key) {
    RedisModule_FreeString(NULL, (RedisModuleString *)key);
}

void *RAI_RStringsKeyDup(void *privdata, const void *key) {
    return RedisModule_CreateStringFromString(NULL, (RedisModuleString *)key);
}
