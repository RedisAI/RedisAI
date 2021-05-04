/**
 * script.c
 *
 * Contains the helper methods for both creating, populating,
 * managing and destructing the PyTorch Script data structure.
 *
 */

#include <pthread.h>
#include "version.h"
#include "script.h"
#include "script_struct.h"
#include "stats.h"
#include "util/arr.h"
#include "util/string_utils.h"
#include "rmutil/alloc.h"
#include "backends/backends.h"
#include "execution/DAG/dag.h"
#include "execution/run_info.h"

extern RedisModuleType *RedisAI_ScriptType;

RAI_Script *RAI_ScriptCreate(const char *devicestr, RedisModuleString *tag, const char *scriptdef,
                             RAI_Error *err) {
    if (!RAI_backends.torch.script_create) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TORCH");
        return NULL;
    }
    RAI_Script *script = RAI_backends.torch.script_create(devicestr, scriptdef, err);

    if (script) {
        if (tag) {
            script->tag = RAI_HoldString(tag);
        } else {
            script->tag = RedisModule_CreateString(NULL, "", 0);
        }
    }

    return script;
}

void RAI_ScriptFree(RAI_Script *script, RAI_Error *err) {
    if (__atomic_sub_fetch(&script->refCount, 1, __ATOMIC_RELAXED) > 0) {
        return;
    }

    if (!RAI_backends.torch.script_free) {
        RAI_SetError(err, RAI_EBACKENDNOTLOADED, "ERR Backend not loaded: TORCH");
        return;
    }

    RedisModule_FreeString(NULL, script->tag);

    RAI_RemoveStatsEntry(script->infokey);

    RAI_backends.torch.script_free(script, err);
}

RAI_Script *RAI_ScriptGetShallowCopy(RAI_Script *script) {
    __atomic_fetch_add(&script->refCount, 1, __ATOMIC_RELAXED);
    return script;
}

/* Return REDISMODULE_ERR if there was an error getting the Script.
 * Return REDISMODULE_OK if the model value stored at key was correctly
 * returned and available at *model variable. */
int RAI_GetScriptFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName, RAI_Script **script,
                              int mode, RAI_Error *err) {
    RedisModuleKey *key = RedisModule_OpenKey(ctx, keyName, mode);

    if (RedisModule_KeyType(key) == REDISMODULE_KEYTYPE_EMPTY) {
        RedisModule_CloseKey(key);
#ifndef LITE
        RedisModule_Log(ctx, "warning", "could not load %s from keyspace, key doesn't exist",
                        RedisModule_StringPtrLen(keyName, NULL));
        RAI_SetError(err, RAI_EKEYEMPTY, "ERR script key is empty");
        return REDISMODULE_ERR;
#else
        if (VerifyKeyInThisShard(ctx, keyName)) { // Relevant for enterprise cluster.
            RAI_SetError(err, RAI_EKEYEMPTY, "ERR script key is empty");
        } else {
            RAI_SetError(err, RAI_EKEYEMPTY,
                         "ERR CROSSSLOT Keys in request don't hash to the same slot");
        }
#endif
        return REDISMODULE_ERR;
    }
    if (RedisModule_ModuleTypeGetType(key) != RedisAI_ScriptType) {
        RedisModule_CloseKey(key);
        RAI_SetError(err, RAI_ESCRIPTRUN, REDISMODULE_ERRORMSG_WRONGTYPE);
        return REDISMODULE_ERR;
    }
    *script = RedisModule_ModuleTypeGetValue(key);
    RedisModule_CloseKey(key);
    return REDISMODULE_OK;
}

int RedisAI_ScriptRun_IsKeysPositionRequest_ReportKeys(RedisModuleCtx *ctx,
                                                       RedisModuleString **argv, int argc) {
    RedisModule_KeyAtPos(ctx, 1);
    size_t startpos = 3;
    if (startpos >= argc) {
        return REDISMODULE_ERR;
    }
    const char *str = RedisModule_StringPtrLen(argv[startpos], NULL);
    if (!strcasecmp(str, "TIMEOUT")) {
        startpos += 2;
    }
    startpos += 1;
    if (startpos >= argc) {
        return REDISMODULE_ERR;
    }
    for (size_t argpos = startpos; argpos < argc; argpos++) {
        str = RedisModule_StringPtrLen(argv[argpos], NULL);
        if (!strcasecmp(str, "OUTPUTS")) {
            continue;
        }
        RedisModule_KeyAtPos(ctx, argpos);
    }
    return REDISMODULE_OK;
}

int RedisAI_ScriptExecute_IsKeysPositionRequest_ReportKeys(RedisModuleCtx *ctx,
                                                           RedisModuleString **argv, int argc) {
    // AI.SCRIPTEXECUTE script_name func KEYS n key....
    if (argc < 6) {
        return REDISMODULE_ERR;
    }
    RedisModule_KeyAtPos(ctx, 1);
    size_t argpos = 3;
    long long count;
    while (argpos < argc) {
        const char *str = RedisModule_StringPtrLen(argv[argpos++], NULL);

        // Inputs, outpus, keys, lists.
        if ((!strcasecmp(str, "INPUTS")) || (!strcasecmp(str, "OUTPUTS")) ||
            (!strcasecmp(str, "LIST_INPUTS")) || (!strcasecmp(str, "KEYS"))) {
            bool updateKeyAtPos = false;
            // The only scope where the inputs strings are 100% keys are in the KEYS and OUTPUTS
            // scopes.
            if ((!strcasecmp(str, "OUTPUTS")) || (!strcasecmp(str, "KEYS"))) {
                updateKeyAtPos = true;
            }
            if (argpos >= argc) {
                return REDISMODULE_ERR;
            }
            if (RedisModule_StringToLongLong(argv[argpos++], &count) != REDISMODULE_OK) {
                return REDISMODULE_ERR;
            }
            if (count <= 0) {
                return REDISMODULE_ERR;
            }
            if (argpos + count >= argc) {
                return REDISMODULE_ERR;
            }
            for (long long i = 0; i < count; i++) {
                if (updateKeyAtPos) {
                    RedisModule_KeyAtPos(ctx, argpos);
                }
                argpos++;
            }
            continue;
        }
        // Timeout
        if (!strcasecmp(str, "TIMEOUT")) {
            argpos++;
            break;
        }
        // Undefinded input.
        return REDISMODULE_ERR;
    }
    if (argpos != argc) {
        return REDISMODULE_ERR;
    } else {
        return REDISMODULE_OK;
    }
}

RedisModuleType *RAI_ScriptRedisType(void) { return RedisAI_ScriptType; }
