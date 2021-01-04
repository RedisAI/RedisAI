#include "util/dict.h"
#include "redismodule.h"
#include "tensor.h"
#include "dag_parser.h"
#include "command_parser.h"
#include "modelRun_ctx.h"
#include "dag_execute.h"
#include <string.h>
#include "dag.h"
#include "string_utils.h"

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
static int _ParseDAGLoadArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                             AI_dict **localContextDict, const char *chaining_operator) {
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
    size_t arg_len;

    // Go over the given args and load the tensors from keyspace.
    for (size_t argpos = 2; argpos < argc && number_loaded_keys < n_keys; argpos++) {
        const char *arg_string = RedisModule_StringPtrLen(argv[argpos], &arg_len);
        if (!strcasecmp(arg_string, chaining_operator))
            break;
        RAI_Tensor *t;
        RedisModuleKey *key;
        const int status = RAI_GetTensorFromKeyspace(ctx, argv[argpos], &key, &t, REDISMODULE_READ);
        if (status == REDISMODULE_ERR) {
            RedisModule_Log(ctx, "warning",
                            "on DAGRUN's LOAD could not load tensor %s from keyspace", arg_string);
            return -1;
        }

        // Add the tensor under its "mangled" key name to the DAG local context dict.
        char buf[16];
        sprintf(buf, "%04d", 1);
        RedisModuleString *dictKey = RedisModule_CreateStringFromString(NULL, argv[argpos]);
        RedisModule_StringAppendBuffer(NULL, dictKey, buf, strlen(buf));
        AI_dictAdd(*localContextDict, (void *)dictKey, (void *)RAI_TensorGetShallowCopy(t));
        RedisModule_FreeString(NULL, dictKey);
        number_loaded_keys++;
    }

    if (number_loaded_keys != n_keys) {
        RedisModule_WrongArity(ctx);
        return -1;
    }
    return number_loaded_keys + 2;
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
static int _ParseDAGPersistArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
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

    // Go over the given args and save the tensor key names to persist.
    int number_keys_to_persist = 0;
    for (size_t argpos = 2; (argpos < argc) && (number_keys_to_persist < n_keys); argpos++) {
        const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
        if (!strcasecmp(arg_string, chaining_operator)) {
            break;
        } else {
            AI_dictAdd(*persistContextDict, (void *)argv[argpos], (void *)1);
            number_keys_to_persist++;
        }
    }
    if (number_keys_to_persist != n_keys) {
        RedisModule_WrongArity(ctx);
        return -1;
    }
    return number_keys_to_persist + 2;
}

static int _parseTimeout(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                         long long *timeout) {

    if (argc == 0) {
        RedisModule_ReplyWithError(ctx, "ERR No value provided for TIMEOUT");
        return REDISMODULE_ERR;
    }
    const int retval = RedisModule_StringToLongLong(argv[1], timeout);
    if (retval != REDISMODULE_OK || timeout <= 0) {
        RedisModule_ReplyWithError(ctx, "ERR Invalid value for TIMEOUT");
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

static RAI_DagOp *_AddEmptyOp(RedisAI_RunInfo *rinfo) {
    RAI_DagOp *currentDagOp;
    RAI_InitDagOp(&currentDagOp);
    rinfo->dagOps = array_append(rinfo->dagOps, currentDagOp);
    return currentDagOp;
}

// Go over the args and save them in the current op until we see another "|>" or finish.
int _CollectOpArgs(RedisModuleString **argv, int argc, int arg_pos, RAI_DagOp *op) {

    op->argv = &argv[arg_pos];
    while (arg_pos < argc) {
        const char *arg_string = RedisModule_StringPtrLen(argv[arg_pos], NULL);
        if (!strcasecmp(arg_string, "|>"))
            return op->argc;
        op->argc++;
        arg_pos++;
    }
    return op->argc;
}

int _ParseDAGOps(RedisModuleCtx *ctx, RedisAI_RunInfo *rinfo) {

    for (long long i = 0; i < array_len(rinfo->dagOps); i++) {
        RAI_DagOp *currentOp = rinfo->dagOps[i];
        // The first op arg is the command name.
        const char *arg_string = RedisModule_StringPtrLen(currentOp->argv[0], NULL);

        if (!strcasecmp(arg_string, "AI.TENSORGET")) {
            currentOp->commandType = REDISAI_DAG_CMD_TENSORGET;
            currentOp->devicestr = "CPU";
            RAI_HoldString(NULL, currentOp->argv[1]);
            currentOp->inkeys = array_append(currentOp->inkeys, currentOp->argv[1]);
            currentOp->fmt = ParseTensorGetArgs(ctx, currentOp->argv, currentOp->argc);
            if (currentOp->fmt == REDISAI_TENSOR_NONE)
                return REDISMODULE_ERR;
            continue;
        }
        if (!strcasecmp(arg_string, "AI.TENSORSET")) {
            currentOp->commandType = REDISAI_DAG_CMD_TENSORSET;
            currentOp->devicestr = "CPU";
            RAI_HoldString(NULL, currentOp->argv[1]);
            currentOp->outkeys = array_append(currentOp->outkeys, currentOp->argv[1]);
            if (RAI_parseTensorSetArgs(ctx, currentOp->argv, currentOp->argc, &currentOp->outTensor,
                                       0, currentOp->err) == -1)
                return REDISMODULE_ERR;
            continue;
        }
        if (!strcasecmp(arg_string, "AI.MODELRUN")) {
            if (ParseModelRunCommand(rinfo, currentOp, ctx, currentOp->argv, currentOp->argc) !=
                REDISMODULE_OK) {
                return REDISMODULE_ERR;
            }
            continue;
        }
        if (!strcasecmp(arg_string, "AI.SCRIPTRUN")) {
            if (ParseScriptRunCommand(rinfo, currentOp, ctx, currentOp->argv, currentOp->argc) !=
                REDISMODULE_OK) {
                return REDISMODULE_ERR;
            }
            continue;
        }
        // If none of the cases match, we have an invalid op.
        RedisModule_ReplyWithError(ctx, "unsupported command within DAG");
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

// Parse the DAG run command and return REDISMODULE_OK only if it is a valid command to execute.
int ParseDAGRunCommand(RedisAI_RunInfo *rinfo, RedisModuleCtx *ctx, RedisModuleString **argv,
                       int argc, bool dag_ro) {

    RAI_Error err = {0};
    if (argc < 4) {
        RedisModule_WrongArity(ctx);
        goto cleanup;
    }

    int chainingOpCount = 0;
    int arg_pos = 1;
    bool load_complete = false;
    bool persist_complete = false;
    bool timeout_complete = false;

    // The first arg is "AI.DAGRUN", so we go over from the next arg.
    while (arg_pos < argc) {
        const char *arg_string = RedisModule_StringPtrLen(argv[arg_pos], NULL);

        if (!strcasecmp(arg_string, "LOAD") && !load_complete && chainingOpCount == 0) {
            /* Load the required tensors from key space and store them in both
               dagTensorsLoadedContext and dagTensorsContext dicts. */
            const int parse_result = _ParseDAGLoadArgs(ctx, &argv[arg_pos], argc - arg_pos,
                                                       &(rinfo->dagTensorsContext), "|>");
            if (parse_result <= 0)
                goto cleanup;
            arg_pos += parse_result;
            load_complete = true;
            continue;
        }
        if (!strcasecmp(arg_string, "PERSIST") && !persist_complete && chainingOpCount == 0) {
            if (dag_ro) {
                RedisModule_ReplyWithError(ctx,
                                           "ERR PERSIST cannot be specified in a read-only DAG");
                goto cleanup;
            }
            /* Store the keys to persist in dagTensorsPersistedContext dict.
               These keys will be populated later on with actual tensors. */
            const int parse_result = _ParseDAGPersistArgs(
                ctx, &argv[arg_pos], argc - arg_pos, &(rinfo->dagTensorsPersistedContext), "|>");

            if (parse_result <= 0)
                goto cleanup;
            arg_pos += parse_result;
            persist_complete = true;
            continue;
        }
        if (!strcasecmp(arg_string, "TIMEOUT") && !timeout_complete && chainingOpCount == 0) {
            long long timeout;
            if (_parseTimeout(ctx, &argv[arg_pos], argc - arg_pos, &timeout) == REDISMODULE_ERR)
                goto cleanup;
            rinfo->timeout = timeout;
            arg_pos += 2;
            timeout_complete = true;
            continue;
        }

        if (!strcasecmp(arg_string, "|>") && arg_pos < argc - 1) {
            RAI_DagOp *currentOp = _AddEmptyOp(rinfo);
            chainingOpCount++;
            int args_num = _CollectOpArgs(argv, argc, ++arg_pos, currentOp);
            arg_pos += args_num;
            continue;
        }
        // If none of the cases match, we have an invalid op.
        RedisModule_ReplyWithError(ctx, "ERR Invalid DAGRUN command");
        goto cleanup;
    }
    rinfo->dagOpCount = array_len(rinfo->dagOps);
    if (rinfo->dagOpCount < 1) {
        RedisModule_ReplyWithError(ctx, "ERR DAG is empty");
        goto cleanup;
    }
    if (_ParseDAGOps(ctx, rinfo) != REDISMODULE_OK)
        goto cleanup;
    if (MangleTensorsNames(rinfo, &err) != REDISMODULE_OK) {
        RedisModule_ReplyWithError(ctx, err.detail_oneline);
        goto cleanup;
    }
    DAG_SetTensorsInLocalContext(rinfo);

    return REDISMODULE_OK;

cleanup:
    RAI_ClearError(&err);
    RAI_FreeRunInfo(rinfo);
    return REDISMODULE_ERR;
}
