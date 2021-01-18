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
                             AI_dict **localContextDict, const char *chaining_operator,
                             RAI_Error *err) {
    if (argc < 3) {
        RAI_SetError(err, RAI_EDAGBUILDER,
                     "ERR wrong number of arguments for LOAD in 'AI.DAGRUN' command");
        return -1;
    }

    long long n_keys;
    const int retval = RedisModule_StringToLongLong(argv[1], &n_keys);
    if (retval != REDISMODULE_OK || n_keys <= 0) {
        RAI_SetError(err, RAI_EDAGBUILDER,
                     "ERR invalid or negative value found in number of keys to LOAD");
        return -1;
    }

    int number_loaded_keys = 0;
    size_t arg_len;

    // Go over the given args and load the tensors from keyspace.
    for (size_t argpos = 2; argpos < argc && number_loaded_keys < n_keys; argpos++) {
        RedisModuleString *key_name = argv[argpos];
        const char *arg_string = RedisModule_StringPtrLen(key_name, &arg_len);
        if (!strcasecmp(arg_string, chaining_operator))
            break;
        RAI_Tensor *t;
        RedisModuleKey *key;
        const int status =
            RAI_GetTensorFromKeyspace(ctx, key_name, &key, &t, REDISMODULE_READ, err);
        if (status == REDISMODULE_ERR) {
            RedisModule_Log(ctx, "warning",
                            "on DAGRUN's LOAD could not load tensor %s from keyspace", arg_string);
            return -1;
        }

        // Add the tensor under its "mangled" key name to the DAG local context dict.
        char buf[16];
        sprintf(buf, "%04d", 1);
        RedisModule_StringAppendBuffer(NULL, key_name, buf, strlen(buf));
        AI_dictAdd(*localContextDict, (void *)key_name, (void *)RAI_TensorGetShallowCopy(t));
        number_loaded_keys++;
    }

    if (number_loaded_keys != n_keys) {
        RAI_SetError(err, RAI_EDAGBUILDER,
                     "ERR number of keys to LOAD in AI.DAGRUN command does not match the number of "
                     "given arguments");
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
                                AI_dict **persistContextDict, const char *chaining_operator,
                                RAI_Error *err) {
    if (argc < 3) {
        RAI_SetError(err, RAI_EDAGBUILDER,
                     "ERR wrong number of arguments for PERSIST in 'AI.DAGRUN' command");
        return -1;
    }

    long long n_keys;
    const int retval = RedisModule_StringToLongLong(argv[1], &n_keys);
    if (retval != REDISMODULE_OK || n_keys <= 0) {
        RAI_SetError(err, RAI_EDAGBUILDER,
                     "ERR invalid or negative value found in number of keys to PERSIST");
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
        RAI_SetError(err, RAI_EDAGBUILDER,
                     "ERR number of keys to PERSIST in AI.DAGRUN command does not match the number "
                     "of given arguments");
        return -1;
    }
    return number_keys_to_persist + 2;
}

static int _parseTimeout(RedisModuleString **argv, int argc, long long *timeout, RAI_Error *err) {

    if (argc == 0) {
        RAI_SetError(err, RAI_EDAGBUILDER, "ERR No value provided for TIMEOUT");
        return REDISMODULE_ERR;
    }
    const int retval = RedisModule_StringToLongLong(argv[1], timeout);
    if (retval != REDISMODULE_OK || timeout <= 0) {
        RAI_SetError(err, RAI_EDAGBUILDER, "ERR Invalid value for TIMEOUT");
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

static RAI_DagOp *_AddEmptyOp(RAI_DagOp ***ops) {
    RAI_DagOp *currentDagOp;
    RAI_InitDagOp(&currentDagOp);
    *ops = array_append(*ops, currentDagOp);
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

int ParseDAGOps(RedisAI_RunInfo *rinfo, RAI_DagOp **ops) {

    for (long long i = 0; i < array_len(ops); i++) {
        RAI_DagOp *currentOp = ops[i];
        // The first op arg is the command name.
        const char *arg_string = RedisModule_StringPtrLen(currentOp->argv[0], NULL);

        if (!strcasecmp(arg_string, "AI.TENSORGET")) {
            currentOp->commandType = REDISAI_DAG_CMD_TENSORGET;
            currentOp->devicestr = "CPU";
            RAI_HoldString(NULL, currentOp->argv[1]);
            currentOp->inkeys = array_append(currentOp->inkeys, currentOp->argv[1]);
            currentOp->fmt = ParseTensorGetArgs(rinfo->err, currentOp->argv, currentOp->argc);
            if (currentOp->fmt == TENSOR_NONE)
                return REDISMODULE_ERR;
            continue;
        }
        if (!strcasecmp(arg_string, "AI.TENSORSET")) {
            currentOp->commandType = REDISAI_DAG_CMD_TENSORSET;
            currentOp->devicestr = "CPU";
            RAI_HoldString(NULL, currentOp->argv[1]);
            currentOp->outkeys = array_append(currentOp->outkeys, currentOp->argv[1]);
            if (RAI_parseTensorSetArgs(currentOp->argv, currentOp->argc, &currentOp->outTensor, 0,
                                       rinfo->err) == -1)
                return REDISMODULE_ERR;
            continue;
        }
        if (!strcasecmp(arg_string, "AI.MODELRUN")) {
            if (ParseModelRunCommand(rinfo, currentOp, currentOp->argv, currentOp->argc) !=
                REDISMODULE_OK) {
                return REDISMODULE_ERR;
            }
            continue;
        }
        if (!strcasecmp(arg_string, "AI.SCRIPTRUN")) {
            if (ParseScriptRunCommand(rinfo, currentOp, currentOp->argv, currentOp->argc) !=
                REDISMODULE_OK) {
                return REDISMODULE_ERR;
            }
            continue;
        }
        // If none of the cases match, we have an invalid op.
        RAI_SetError(rinfo->err, RAI_EDAGBUILDER, "unsupported command within DAG");
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

int ParseDAGRunCommand(RedisAI_RunInfo *rinfo, RedisModuleCtx *ctx, RedisModuleString **argv,
                       int argc, bool dag_ro) {

    int res = REDISMODULE_ERR;
    if (argc < 4) {
        if (dag_ro) {
            RAI_SetError(rinfo->err, RAI_EDAGBUILDER,
                         "ERR wrong number of arguments for 'AI.DAGRUN_RO' command");
        } else {
            RAI_SetError(rinfo->err, RAI_EDAGBUILDER,
                         "ERR wrong number of arguments for 'AI.DAGRUN' command");
        }
        return res;
    }

    int chainingOpCount = 0;
    int arg_pos = 1;
    bool load_complete = false;
    bool persist_complete = false;
    bool timeout_complete = false;
    array_new_on_stack(RAI_DagOp *, 10, dag_ops);

    // The first arg is "AI.DAGRUN", so we go over from the next arg.
    while (arg_pos < argc) {
        const char *arg_string = RedisModule_StringPtrLen(argv[arg_pos], NULL);

        if (!strcasecmp(arg_string, "LOAD") && !load_complete && chainingOpCount == 0) {
            /* Load the required tensors from key space and store them in both
               dagTensorsLoadedContext and dagTensorsContext dicts. */
            const int parse_result = _ParseDAGLoadArgs(
                ctx, &argv[arg_pos], argc - arg_pos, &(rinfo->dagTensorsContext), "|>", rinfo->err);
            if (parse_result <= 0)
                goto cleanup;
            arg_pos += parse_result;
            load_complete = true;
            continue;
        }
        if (!strcasecmp(arg_string, "PERSIST") && !persist_complete && chainingOpCount == 0) {
            if (dag_ro) {
                RAI_SetError(rinfo->err, RAI_EDAGBUILDER,
                             "ERR PERSIST cannot be specified in a read-only DAG");
                goto cleanup;
            }
            /* Store the keys to persist in dagTensorsPersistedContext dict.
               These keys will be populated later on with actual tensors. */
            const int parse_result =
                _ParseDAGPersistArgs(ctx, &argv[arg_pos], argc - arg_pos,
                                     &(rinfo->dagTensorsPersistedContext), "|>", rinfo->err);

            if (parse_result <= 0)
                goto cleanup;
            arg_pos += parse_result;
            persist_complete = true;
            continue;
        }
        if (!strcasecmp(arg_string, "TIMEOUT") && !timeout_complete && chainingOpCount == 0) {
            long long timeout;
            if (_parseTimeout(&argv[arg_pos], argc - arg_pos, &timeout, rinfo->err) ==
                REDISMODULE_ERR)
                goto cleanup;
            rinfo->timeout = timeout;
            arg_pos += 2;
            timeout_complete = true;
            continue;
        }

        if (!strcasecmp(arg_string, "|>") && arg_pos < argc - 1) {
            RAI_DagOp *currentOp = _AddEmptyOp(&dag_ops);
            chainingOpCount++;
            int args_num = _CollectOpArgs(argv, argc, ++arg_pos, currentOp);
            arg_pos += args_num;
            continue;
        }
        // If none of the cases match, we have an invalid op.
        RAI_SetError(rinfo->err, RAI_EDAGBUILDER, "ERR Invalid DAGRUN command");
        goto cleanup;
    }

    if (array_len(dag_ops) < 1) {
        RAI_SetError(rinfo->err, RAI_EDAGBUILDER, "ERR DAG is empty");
        goto cleanup;
    }

    if (ParseDAGOps(rinfo, dag_ops) != REDISMODULE_OK) {
        for (size_t i = 0; i < array_len(dag_ops); i++) {
            RAI_FreeDagOp(dag_ops[i]);
        }
        goto cleanup;
    }
    // After validating all the ops, insert them to the DAG.
    for (size_t i = 0; i < array_len(dag_ops); i++) {
        rinfo->dagOps = array_append(rinfo->dagOps, dag_ops[i]);
    }
    rinfo->dagOpCount = array_len(rinfo->dagOps);

    if (MangleTensorsNames(rinfo) != REDISMODULE_OK) {
        goto cleanup;
    }
    res = REDISMODULE_OK;

cleanup:
    array_free(dag_ops);
    return res;
}
