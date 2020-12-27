#include "util/dict.h"
#include "redismodule.h"
#include "tensor.h"
#include "dag_parser.h"
#include "modelRun_ctx.h"
#include <string.h>
#include "util/string_utils.h"

/**
 * DAGRUN Building Block to parse [LOAD <nkeys> key1 key2... ]
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @param loadedContextDict local non-blocking hash table containing key names
 * loaded from the keyspace tensors
 * @param localContextDict local non-blocking hash table containing DAG's
 * tensors
 * @param chaining_operator operator used to split operations. Any command
 * argument after the chaining operator is not considered
 * @return processed number of arguments on success, or -1 if the parsing failed
 */
static int DAG_ParseLoadArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                             AI_dict **loadedContextDict, AI_dict **localContextDict,
                             const char *chaining_operator) {
    if (argc < 3) {
        RedisModule_WrongArity(ctx);
        return -1;
    }

    long long n_keys;
    const int retval = RedisModule_StringToLongLong(argv[1], &n_keys);
    if (retval != REDISMODULE_OK || n_keys <= 0) {
        RedisModule_ReplyWithError(ctx,
                                   "ERR invalid or negative value found in number of keys to LOAD");
        return -1;
    }
    int number_loaded_keys = 0;
    int separator_flag = 0;
    size_t argpos = 2;
    for (; (argpos <= argc - 1) && (number_loaded_keys < n_keys); argpos++) {
        size_t arg_len;
        const char *arg_string = RedisModule_StringPtrLen(argv[argpos], &arg_len);
        if (!strcasecmp(arg_string, chaining_operator)) {
            separator_flag = 1;
            break;
        } else {
            RAI_Tensor *t;
            RedisModuleKey *key;
            const int status =
                RAI_GetTensorFromKeyspace(ctx, argv[argpos], &key, &t, REDISMODULE_READ);
            if (status == REDISMODULE_ERR) {
                RedisModule_Log(ctx, "warning",
                                "on DAGRUN's LOAD could not load tensor %s from keyspace",
                                arg_string);
                return -1;
            }
            char buf[16];
            sprintf(buf, "%04d", 1);
            RedisModuleString *dictKey = RedisModule_CreateStringFromString(NULL, argv[argpos]);
            RedisModule_StringAppendBuffer(NULL, dictKey, buf, strlen(buf));

            AI_dictAdd(*localContextDict, (void *)dictKey, (void *)RAI_TensorGetShallowCopy(t));
            AI_dictAdd(*loadedContextDict, (void *)dictKey, (void *)1);
            RedisModule_FreeString(NULL, dictKey);
            number_loaded_keys++;
        }
    }
    if (number_loaded_keys != n_keys) {
        RedisModule_WrongArity(ctx);
        return -1;
    }
    return argpos;
}

/**
 * DAGRUN Building Block to parse [PERSIST <nkeys> key1 key2... ]
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @param localContextDict local non-blocking hash table containing DAG's
 * keynames marked as persistent
 * @param chaining_operator operator used to split operations. Any command
 * argument after the chaining operator is not considered
 * @return processed number of arguments on success, or -1 if the parsing failed
 */
static int DAG_ParsePersistArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                                AI_dict **persistContextDict, const char *chaining_operator) {
    if (argc < 3) {
        RedisModule_WrongArity(ctx);
        return -1;
    }

    long long n_keys;
    const int retval = RedisModule_StringToLongLong(argv[1], &n_keys);
    if (retval != REDISMODULE_OK || n_keys <= 0) {
        RedisModule_ReplyWithError(
            ctx, "ERR invalid or negative value found in number of keys to PERSIST");
        return -1;
    }

    int number_loaded_keys = 0;
    int separator_flag = 0;
    size_t argpos = 2;
    for (; (argpos < argc) && (number_loaded_keys < n_keys); argpos++) {
        const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
        if (!strcasecmp(arg_string, chaining_operator)) {
            separator_flag = 1;
            break;
        } else {
            AI_dictAdd(*persistContextDict, (void *)argv[argpos], (void *)1);
            number_loaded_keys++;
        }
    }
    if (number_loaded_keys != n_keys) {
        RedisModule_WrongArity(ctx);
        return -1;
    }
    return argpos;
}

// Parse the DAG run command and return REDISMODULE_OK only if it is a valid command to execute.
int DAG_CommandParser(RedisModuleCtx *ctx, RedisModuleString **argv, int argc, bool dag_ro,
                      RedisAI_RunInfo **rinfo_ptr) {

    if (argc < 4) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_ERR;
    }
    RedisAI_RunInfo *rinfo = *rinfo_ptr;
    RAI_DagOp *currentDagOp = NULL;
    RAI_InitDagOp(&currentDagOp);
    rinfo->dagOps = array_append(rinfo->dagOps, currentDagOp);

    int chainingOpCount = 0;
    bool load_complete = false;
    bool persist_complete = false;

    // The first arg is "AI.DAGRUN", so we go over from the next arg.
    for (int arg_pos = 1; arg_pos < argc; arg_pos++) {
        const char *arg_string = RedisModule_StringPtrLen(argv[arg_pos], NULL);

        if (!strcasecmp(arg_string, "LOAD") && !load_complete) {
            /* Load the required tensors from key space and store them in both
               dagTensorsLoadedContext and dagTensorsContext dicts. */
            const int parse_result = DAG_ParseLoadArgs(ctx, &argv[arg_pos], argc - arg_pos,
                                                       &(rinfo->dagTensorsLoadedContext),
                                                       &(rinfo->dagTensorsContext), "|>");
            if (parse_result > 0) {
                arg_pos += parse_result - 1;
                load_complete = true;
            } else {
                RAI_FreeRunInfo(rinfo);
                return REDISMODULE_ERR;
            }
        } else if (!strcasecmp(arg_string, "PERSIST") && !persist_complete) {
            if (dag_ro) {
                RAI_FreeRunInfo(rinfo);
                RedisModule_ReplyWithError(ctx,
                                           "ERR PERSIST cannot be specified in a read-only DAG");
                return REDISMODULE_ERR;
            }
            /* Store the keys to persist in dagTensorsPersistedContext dict.
               These keys will be populated late on with actual tensors. */
            const int parse_result = DAG_ParsePersistArgs(
                ctx, &argv[arg_pos], argc - arg_pos, &(rinfo->dagTensorsPersistedContext), "|>");
            if (parse_result > 0) {
                arg_pos += parse_result - 1;
                persist_complete = true;
            } else {
                RAI_FreeRunInfo(rinfo);
                return REDISMODULE_ERR;
            }
        } else if (!strcasecmp(arg_string, "TIMEOUT")) {
            if (!((chainingOpCount == 0) || (chainingOpCount == 1 && rinfo->single_op_dag == 1))) {
                RAI_FreeRunInfo(rinfo);
                RedisModule_ReplyWithError(ctx, "ERR TIMEOUT not allowed within a DAG command");
                return REDISMODULE_ERR;
            }
            if (arg_pos == argc - 1) {
                RAI_FreeRunInfo(rinfo);
                RedisModule_ReplyWithError(ctx, "ERR No value provided for TIMEOUT");
                return REDISMODULE_ERR;
            }
            long long timeout;
            const int retval = RedisModule_StringToLongLong(argv[arg_pos + 1], &timeout);
            if (retval != REDISMODULE_OK || timeout <= 0) {
                RAI_FreeRunInfo(rinfo);
                RedisModule_ReplyWithError(ctx, "ERR Invalid value for TIMEOUT");
                return REDISMODULE_ERR;
            }
            rinfo->timeout = timeout;
            arg_pos += 1;
            continue;
        } else if (!strcasecmp(arg_string, "|>") && arg_pos < argc - 1) {
            // on the first pipe operator, if LOAD or PERSIST were used, we've already
            // allocated memory
            if (chainingOpCount > 0) {
                rinfo->dagOpCount++;
                RAI_DagOp *currentDagOp = NULL;
                RAI_InitDagOp(&currentDagOp);
                rinfo->dagOps = array_append(rinfo->dagOps, currentDagOp);
            }
            chainingOpCount++;
        } else {
            if (!strcasecmp(arg_string, "AI.TENSORGET")) {
                rinfo->dagOps[rinfo->dagOpCount]->commandType = REDISAI_DAG_CMD_TENSORGET;
                rinfo->dagOps[rinfo->dagOpCount]->devicestr = "CPU";
            }
            if (!strcasecmp(arg_string, "AI.TENSORSET")) {
                rinfo->dagOps[rinfo->dagOpCount]->commandType = REDISAI_DAG_CMD_TENSORSET;
                rinfo->dagOps[rinfo->dagOpCount]->devicestr = "CPU";
            }
            if (!strcasecmp(arg_string, "AI.MODELRUN")) {
                if (argc - 2 < arg_pos) {
                    RedisModule_WrongArity(ctx);
                    return REDISMODULE_ERR;
                }
                RAI_DagOp *currentOp = rinfo->dagOps[rinfo->dagOpCount];
                currentOp->commandType = REDISAI_DAG_CMD_MODELRUN;
                RAI_Model *mto;
                RedisModuleKey *modelKey;
                const int status = RAI_GetModelFromKeyspace(ctx, argv[arg_pos + 1], &modelKey, &mto,
                                                            REDISMODULE_READ);
                RedisModule_OpenKey(ctx, argv[arg_pos + 1], REDISMODULE_READ);
                if (status == REDISMODULE_ERR) {
                    RAI_FreeRunInfo(rinfo);
                    RedisModule_ReplyWithError(ctx, "ERR Model not found");
                    return REDISMODULE_ERR;
                }
                currentOp->devicestr = mto->devicestr;
                currentOp->runkey = argv[arg_pos + 1];
                currentOp->mctx = RAI_ModelRunCtxCreate(mto);
            }
            if (!strcasecmp(arg_string, "AI.SCRIPTRUN")) {
                if (argc - 3 < arg_pos) {
                    RedisModule_WrongArity(ctx);
                    return REDISMODULE_ERR;
                }
                RAI_DagOp *currentOp = rinfo->dagOps[rinfo->dagOpCount];
                currentOp->commandType = REDISAI_DAG_CMD_SCRIPTRUN;
                RAI_Script *sto;
                RedisModuleKey *scriptKey;
                const int status = RAI_GetScriptFromKeyspace(ctx, argv[arg_pos + 1], &scriptKey,
                                                             &sto, REDISMODULE_READ);
                RedisModule_OpenKey(ctx, argv[arg_pos + 1], REDISMODULE_READ);
                if (status == REDISMODULE_ERR) {
                    RAI_FreeRunInfo(rinfo);
                    return REDISMODULE_ERR;
                }
                currentOp->devicestr = sto->devicestr;
                const char *functionName = RedisModule_StringPtrLen(argv[arg_pos + 2], NULL);
                currentOp->runkey = argv[arg_pos + 1];
                currentOp->sctx = RAI_ScriptRunCtxCreate(sto, functionName);
            }
            RAI_HoldString(NULL, argv[arg_pos]);
            RAI_DagOp *currentOp = rinfo->dagOps[rinfo->dagOpCount];
            currentOp->argv = array_append(currentOp->argv, argv[arg_pos]);
            currentOp->argc++;
        }
    }

    rinfo->dagOpCount = array_len(rinfo->dagOps);

    for (long long i = 0; i < array_len(rinfo->dagOps); i++) {
        RAI_DagOp *currentOp = rinfo->dagOps[i];
        if (currentOp == NULL)
            continue;
        int parse_result;
        switch (currentOp->commandType) {
        case REDISAI_DAG_CMD_TENSORSET:
            currentOp->outkeys = array_append(currentOp->outkeys, currentOp->argv[1]);
            break;
        case REDISAI_DAG_CMD_TENSORGET:
            currentOp->inkeys = array_append(currentOp->inkeys, currentOp->argv[1]);
            break;
        case REDISAI_DAG_CMD_MODELRUN:
            parse_result = RedisAI_Parse_ModelRun_RedisCommand(
                NULL, currentOp->argv, currentOp->argc, &(currentOp->mctx), &(currentOp->inkeys),
                &(currentOp->outkeys), &(currentOp->mctx->model), currentOp->err);
            if (parse_result < 0) {
                RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
                return REDISMODULE_ERR;
            }
            break;
        case REDISAI_DAG_CMD_SCRIPTRUN:
            parse_result = RedisAI_Parse_ScriptRun_RedisCommand(
                NULL, currentOp->argv, currentOp->argc, &(currentOp->inkeys), &(currentOp->outkeys),
                &(currentOp->sctx->variadic), currentOp->err);
            if (parse_result < 0) {
                RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
                return REDISMODULE_ERR;
            }
            break;
        }
    }

    // At this point, we have built a sequence of DAG operations, each with its own
    // input and output keys. The names of the keys will be used to look whether the
    // inputs to a DAG operation have all been realized by previous operations (or if
    // they are available as part of LOADed keys from keyspace).
    // This strategy is fine if keys are not aliased, that is, if a command's output
    // overwrites the key of a previous command. This would trick DAG operations into
    // thinking that their input is ready when it's not.
    // To overcome this, we make key names unique, so that names are not aliased. We
    // mangle the names by appending a numerical suffix ":0001". After computing, we
    // demangle the keys in order to persist them.

    AI_dict *mangled_tensors = AI_dictCreate(&AI_dictTypeHeapRStrings, NULL);
    if (!mangled_tensors) {
        return REDISMODULE_ERR;
    }

    {
        AI_dictIterator *iter = AI_dictGetSafeIterator(rinfo->dagTensorsLoadedContext);
        AI_dictEntry *entry = AI_dictNext(iter);
        while (entry) {
            RedisModuleString *key = (RedisModuleString *)AI_dictGetKey(entry);
            size_t key_len;
            const char *key_str = RedisModule_StringPtrLen(key, &key_len);
            RedisModuleString *demangled_key = RedisModule_CreateString(NULL, key_str, key_len - 4);
            int *instance = RedisModule_Alloc(sizeof(int));
            *instance = 1;
            AI_dictAdd(mangled_tensors, (void *)demangled_key, (void *)instance);
            RedisModule_FreeString(NULL, demangled_key);
            entry = AI_dictNext(iter);
        }
        AI_dictReleaseIterator(iter);
    }

    for (long long i = 0; i < array_len(rinfo->dagOps); i++) {
        RAI_DagOp *currentOp = rinfo->dagOps[i];

        RedisModuleString **mangled_inkeys =
            array_new(RedisModuleString *, array_len(currentOp->inkeys));
        for (long long j = 0; j < array_len(currentOp->inkeys); j++) {
            RedisModuleString *key = currentOp->inkeys[j];
            AI_dictEntry *entry = AI_dictFind(mangled_tensors, key);
            if (!entry) {
                AI_dictRelease(mangled_tensors);
                RedisModule_ReplyWithError(ctx, "ERR INPUT key cannot be found in DAG");
                return REDISMODULE_ERR;
            }
            int *instance = AI_dictGetVal(entry);
            char buf[16];
            sprintf(buf, "%04d", *instance);
            RedisModuleString *mangled_key = RedisModule_CreateStringFromString(NULL, key);
            RedisModule_StringAppendBuffer(NULL, mangled_key, buf, strlen(buf));
            mangled_inkeys = array_append(mangled_inkeys, mangled_key);
        }

        RedisModuleString **mangled_outkeys =
            array_new(RedisModuleString *, array_len(currentOp->outkeys));
        for (long long j = 0; j < array_len(currentOp->outkeys); j++) {
            RedisModuleString *key = currentOp->outkeys[j];
            AI_dictEntry *entry = AI_dictFind(mangled_tensors, key);
            int *instance = NULL;
            if (entry) {
                instance = AI_dictGetVal(entry);
                *instance += 1;
            } else {
                instance = RedisModule_Alloc(sizeof(int));
                *instance = 1;
                AI_dictAdd(mangled_tensors, (void *)key, (void *)instance);
            }
            char buf[16];
            sprintf(buf, "%04d", *instance);
            RedisModuleString *mangled_key = RedisModule_CreateStringFromString(NULL, key);
            RedisModule_StringAppendBuffer(NULL, mangled_key, buf, strlen(buf));
            mangled_outkeys = array_append(mangled_outkeys, mangled_key);
        }

        array_free(currentOp->inkeys);
        array_free(currentOp->outkeys);

        currentOp->inkeys = mangled_inkeys;
        currentOp->outkeys = mangled_outkeys;
    }

    AI_dict *mangled_persisted = AI_dictCreate(&AI_dictTypeHeapRStrings, NULL);
    {
        AI_dictIterator *iter = AI_dictGetSafeIterator(rinfo->dagTensorsPersistedContext);
        AI_dictEntry *entry = AI_dictNext(iter);
        while (entry) {
            RedisModuleString *key = (RedisModuleString *)AI_dictGetKey(entry);
            AI_dictEntry *mangled_entry = AI_dictFind(mangled_tensors, key);
            if (!mangled_entry) {
                AI_dictRelease(mangled_tensors);
                AI_dictRelease(mangled_persisted);
                AI_dictReleaseIterator(iter);
                RedisModule_ReplyWithError(ctx, "ERR PERSIST key cannot be found in DAG");
                return REDISMODULE_ERR;
            }
            int *instance = AI_dictGetVal(mangled_entry);
            char buf[16];
            sprintf(buf, "%04d", *instance);
            RedisModuleString *mangled_key = RedisModule_CreateStringFromString(NULL, key);
            RedisModule_StringAppendBuffer(NULL, mangled_key, buf, strlen(buf));

            AI_dictAdd(mangled_persisted, (void *)mangled_key, (void *)1);
            entry = AI_dictNext(iter);
        }
        AI_dictReleaseIterator(iter);
    }

    AI_dictRelease(rinfo->dagTensorsPersistedContext);
    rinfo->dagTensorsPersistedContext = mangled_persisted;

    {
        AI_dictIterator *iter = AI_dictGetSafeIterator(mangled_tensors);
        AI_dictEntry *entry = AI_dictNext(iter);
        while (entry) {
            int *val = (int *)AI_dictGetVal(entry);
            RedisModule_Free(val);
            entry = AI_dictNext(iter);
        }
        AI_dictReleaseIterator(iter);
    }
    AI_dictRelease(mangled_tensors);
    mangled_tensors = NULL;

    for (long long i = 0; i < array_len(rinfo->dagOps); i++) {
        if (rinfo->dagOps[i]->devicestr == NULL) {
            rinfo->dagOps[i]->devicestr = "CPU";
        }
    }
    return REDISMODULE_OK;
}
