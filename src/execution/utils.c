#include "utils.h"
#include "redis_ai_objects/tensor.h"
#include "redis_ai_objects/model.h"

int redisMajorVersion;
int redisMinorVersion;
int redisPatchVersion;

int rlecMajorVersion;
int rlecMinorVersion;
int rlecPatchVersion;
int rlecBuild;

void getRedisVersion() {
    RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(NULL);
    RedisModuleCallReply *reply = RedisModule_Call(ctx, "info", "c", "server");
    assert(RedisModule_CallReplyType(reply) == REDISMODULE_REPLY_STRING);
    size_t len;
    const char *replyStr = RedisModule_CallReplyStringPtr(reply, &len);

    int n = sscanf(replyStr, "# Server\nredis_version:%d.%d.%d", &redisMajorVersion,
                   &redisMinorVersion, &redisPatchVersion);

    assert(n == 3);

    rlecMajorVersion = -1;
    rlecMinorVersion = -1;
    rlecPatchVersion = -1;
    rlecBuild = -1;
    char *enterpriseStr = strstr(replyStr, "rlec_version:");
    if (enterpriseStr) {
        n = sscanf(enterpriseStr, "rlec_version:%d.%d.%d-%d", &rlecMajorVersion, &rlecMinorVersion,
                   &rlecPatchVersion, &rlecBuild);
        if (n != 4) {
            RedisModule_Log(NULL, "warning", "Could not extract enterprise version");
        }
    }

    RedisModule_FreeCallReply(reply);
    RedisModule_FreeThreadSafeContext(ctx);
}

bool IsEnterprise() { return rlecMajorVersion != -1; }

bool VerifyKeyInThisShard(RedisModuleCtx *ctx, RedisModuleString *key_str) {
    if (IsEnterprise()) {
        int key_slot = RedisModule_ShardingGetKeySlot(key_str);
        // If key_slot equals -1, then sharding is not enabled in enterprise.
        if (key_slot != -1) {
            int first_slot, last_slot;
            RedisModule_ShardingGetSlotRange(&first_slot, &last_slot);
            if (key_slot < first_slot || key_slot > last_slot) {
                RedisModule_Log(ctx, "warning",
                                "could not load %s from keyspace,"
                                " this key's hash slot belongs to a different shard",
                                RedisModule_StringPtrLen(key_str, NULL));
                return false;
            }
        }
    }
    return true;
}
