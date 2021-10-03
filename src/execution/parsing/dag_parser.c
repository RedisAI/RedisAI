#include <string.h>
#include "dag_parser.h"
#include "redismodule.h"
#include "util/dict.h"
#include "util/string_utils.h"
#include "execution/execution_contexts/modelRun_ctx.h"
#include "execution/command_parser.h"
#include "execution/DAG/dag.h"
#include "execution/DAG/dag_execute.h"
#include "execution/parsing/deprecated.h"
#include "execution/parsing/tensor_commands_parsing.h"
#include "execution/utils.h"
#include "model_commands_parser.h"
#include "script_commands_parser.h"
#include "parse_utils.h"

/**
 * DAGRUN Building Block to parse [LOAD <nkeys> key1 key2... ]
 *
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @param tensorsToInd Hash table that maps tensor key name to its index in the
 * shared tensors array of the DAG.
 * @param sharedTensors An array that use to store intermideate tensors in the DAG
 * @param chaining_operator operator used to split operations. Any command
 * argument after the chaining operator is not considered
 * @return processed number of arguments on success, or -1 if the parsing failed
 */
static int _ParseDAGLoadArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                             AI_dict *tensorsToInd, RAI_Tensor ***sharedTensors, RAI_Error *err) {
    if (argc < 3) {
        RAI_SetError(err, RAI_EDAGBUILDER,
                     "ERR missing arguments after LOAD keyword in DAG command");
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
        RAI_Tensor *t;
        RedisModuleKey *key;
        int status = RAI_TensorGetFromKeyspace(ctx, key_name, &key, &t, REDISMODULE_READ, err);
        if (status == REDISMODULE_ERR) {
            RedisModule_Log(ctx, "warning", "Could not LOAD tensor %s from keyspace into DAG",
                            arg_string);
            return -1;
        }

        // Add the tensor to the DAG shared tensors and map its name to the relevant index.
        size_t index = array_len(*sharedTensors);
        AI_dictAdd(tensorsToInd, (void *)key_name, (void *)index);
        *sharedTensors = array_append(*sharedTensors, (void *)RAI_TensorGetShallowCopy(t));
        number_loaded_keys++;
    }

    if (number_loaded_keys != n_keys) {
        RAI_SetError(err, RAI_EDAGBUILDER,
                     "ERR number of keys to LOAD into DAG does not match the number of "
                     "given arguments");
        return -1;
    }
    return number_loaded_keys + 2;
}

/**
 * DAGRUN Building Block to parse [PERSIST <nkeys> key1 key2... ]
 * @param ctx Context in which Redis modules operate
 * @param argv Redis command arguments, as an array of strings
 * @param argc Redis command number of arguments
 * @param persistTensorsNames local hash table containing DAG's
 * keynames marked as persistent
 * @param chaining_operator operator used to split operations. Any command
 * argument after the chaining operator is not considered
 * @return processed number of arguments on success, or -1 if the parsing failed
 */
static int _ParseDAGPersistArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                                AI_dict *persistTensorsNames, RAI_Error *err) {
    if (argc < 3) {
        RAI_SetError(err, RAI_EDAGBUILDER,
                     "ERR missing arguments after PERSIST keyword in DAG command");
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
        if (AI_dictFind(persistTensorsNames, (void *)argv[argpos]) != NULL) {
            RAI_SetError(err, RAI_EDAGBUILDER, "ERR PERSIST keys must be unique");
            return -1;
        }
        if (!VerifyKeyInThisShard(ctx, argv[argpos])) { // Relevant for enterprise cluster.
            RAI_SetError(
                err, RAI_EDAGBUILDER,
                "ERR Found keys to persist in DAG command that don't hash to the local shard");
            return -1;
        }
        AI_dictAdd(persistTensorsNames, (void *)argv[argpos], NULL);
        number_keys_to_persist++;
    }
    if (number_keys_to_persist != n_keys) {
        RAI_SetError(err, RAI_EDAGBUILDER,
                     "ERR number of keys to PERSIST after DAG execution does not match the number "
                     "of given arguments");
        return -1;
    }
    return number_keys_to_persist + 2;
}

static int _parseTimeout(RedisModuleString **argv, int argc, long long *timeout, RAI_Error *err) {

    if (argc < 2) {
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

int ParseDAGExecuteOps(RedisAI_RunInfo *rinfo, RAI_DagOp **ops, bool ro) {

    for (long long i = 0; i < array_len(ops); i++) {
        RAI_DagOp *currentOp = ops[i];
        // The first op arg is the command name.
        const char *arg_string = RedisModule_StringPtrLen(currentOp->argv[0], NULL);

        if (!strcasecmp(arg_string, "AI.TENSORGET")) {
            currentOp->commandType = REDISAI_DAG_CMD_TENSORGET;
            currentOp->fmt = ParseTensorGetFormat(rinfo->err, currentOp->argv, currentOp->argc);
            if (currentOp->fmt == TENSOR_NONE) {
                return REDISMODULE_ERR;
            }
            currentOp->devicestr = "CPU";
            RAI_HoldString(currentOp->argv[1]);
            currentOp->inkeys = array_append(currentOp->inkeys, currentOp->argv[1]);
            continue;
        }
        if (!strcasecmp(arg_string, "AI.TENSORSET")) {
            currentOp->commandType = REDISAI_DAG_CMD_TENSORSET;
            if (ParseTensorSetArgs(currentOp->argv, currentOp->argc, &currentOp->outTensor,
                                   rinfo->err) != REDISMODULE_OK) {
                return REDISMODULE_ERR;
            }
            currentOp->devicestr = "CPU";
            RAI_HoldString(currentOp->argv[1]);
            currentOp->outkeys = array_append(currentOp->outkeys, currentOp->argv[1]);
            currentOp->result = REDISMODULE_OK;
            continue;
        }
        if (!strcasecmp(arg_string, "AI.MODELEXECUTE")) {
            if (ParseModelExecuteCommand(rinfo, currentOp, currentOp->argv, currentOp->argc) !=
                REDISMODULE_OK) {
                return REDISMODULE_ERR;
            }
            continue;
        }
        if (!strcasecmp(arg_string, "AI.SCRIPTEXECUTE")) {
            if (ro) {
                // Scripts can contain call to Redis commands (that may write to Redis)
                RAI_SetError(rinfo->err, RAI_EDAGBUILDER,
                             "ERR AI.SCRIPTEXECUTE command cannot be specified in a read-only DAG");
                return REDISMODULE_ERR;
            }
            if (ParseScriptExecuteCommand(rinfo, currentOp, currentOp->argv, currentOp->argc) !=
                REDISMODULE_OK) {
                return REDISMODULE_ERR;
            }
            continue;
        }
        if (!strcasecmp(arg_string, "AI.MODELRUN")) {
            RAI_SetError(rinfo->err, RAI_EDAGBUILDER,
                         "Deprecated AI.MODELRUN"
                         " cannot be used in AI.DAGEXECUTE command");
            return REDISMODULE_ERR;
        }
        if (!strcasecmp(arg_string, "AI.SCRIPTRUN")) {
            RAI_SetError(rinfo->err, RAI_EDAGBUILDER,
                         "Deprecated AI.SCRIPTRUN"
                         " cannot be used in AI.DAGEXECUTE command");
            return REDISMODULE_ERR;
        }
        // If none of the cases match, we have an invalid op.
        RAI_SetError(rinfo->err, RAI_EDAGBUILDER, "Unsupported command within DAG");
        return REDISMODULE_ERR;
    }

    // After validating all the ops, insert them to the DAG.
    for (size_t i = 0; i < array_len(ops); i++) {
        rinfo->dagOps = array_append(rinfo->dagOps, ops[i]);
    }
    rinfo->dagOpCount = array_len(rinfo->dagOps);
    return REDISMODULE_OK;
}

int DAGInitialParsing(RedisAI_RunInfo *rinfo, RedisModuleCtx *ctx, RedisModuleString **argv,
                      int argc, bool dag_ro, RAI_DagOp ***dag_ops) {

    int chainingOpCount = 0;
    int arg_pos = 1;
    bool load_complete = false;
    bool persist_complete = false;
    bool timeout_complete = false;
    bool routing_complete = false;

    // The first arg is "AI.DAGEXECUTE(_RO) (or deprecated AI.DAGRUN(_RO))", so we go over from the
    // next arg.
    while (arg_pos < argc) {
        const char *arg_string = RedisModule_StringPtrLen(argv[arg_pos], NULL);
        if (!strcasecmp(arg_string, "LOAD") && !load_complete && chainingOpCount == 0) {
            /* Load the required tensors from key space to the dag shared tensors
             * array, and save a mapping of their names to the corresponding indices. */
            const int parse_result =
                _ParseDAGLoadArgs(ctx, &argv[arg_pos], argc - arg_pos, rinfo->tensorsNamesToIndices,
                                  &rinfo->dagSharedTensors, rinfo->err);
            if (parse_result <= 0)
                return REDISMODULE_ERR;
            arg_pos += parse_result;
            load_complete = true;
            continue;
        }
        if (!strcasecmp(arg_string, "PERSIST") && !persist_complete && chainingOpCount == 0) {
            if (dag_ro) {
                RAI_SetError(rinfo->err, RAI_EDAGBUILDER,
                             "ERR PERSIST cannot be specified in a read-only DAG");
                return REDISMODULE_ERR;
            }
            /* Store the keys to persist in persistTensors dict, these keys will
             * be mapped later to the indices in the dagSharedTensors array in which the
             * tensors to persist will be found by the end of the DAG run. */
            const int parse_result = _ParseDAGPersistArgs(ctx, &argv[arg_pos], argc - arg_pos,
                                                          rinfo->persistTensors, rinfo->err);
            if (parse_result <= 0)
                return REDISMODULE_ERR;
            arg_pos += parse_result;
            persist_complete = true;
            continue;
        }
        if (!strcasecmp(arg_string, "ROUTING") && !routing_complete && chainingOpCount == 0) {
            arg_pos++;
            if (arg_pos == argc) {
                RAI_SetError(rinfo->err, RAI_EDAGBUILDER, "ERR Missing ROUTING value");
                return REDISMODULE_ERR;
            }
            if (!VerifyKeyInThisShard(ctx, argv[arg_pos++])) {
                RAI_SetError(
                    rinfo->err, RAI_EDAGBUILDER,
                    "ERR ROUTING value specified in the command hash to slot which does not "
                    "belong to the current shard");
                return REDISMODULE_ERR;
            }
            routing_complete = true;
            continue;
        }
        if (!strcasecmp(arg_string, "TIMEOUT") && !timeout_complete && chainingOpCount == 0) {
            long long timeout;
            if (_parseTimeout(&argv[arg_pos], argc - arg_pos, &timeout, rinfo->err) ==
                REDISMODULE_ERR)
                return REDISMODULE_ERR;
            rinfo->timeout = timeout;
            arg_pos += 2;
            timeout_complete = true;
            continue;
        }
        if (!strcasecmp(arg_string, "|>") && arg_pos < argc - 1) {
            RAI_DagOp *currentOp = _AddEmptyOp(dag_ops);
            chainingOpCount++;
            int args_num = _CollectOpArgs(argv, argc, ++arg_pos, currentOp);
            arg_pos += args_num;
            continue;
        }
        // If none of the cases match, we have an invalid op.
        size_t error_len =
            strlen("ERR Invalid DAG command. Unexpected argument: ") + strlen(arg_string) + 1;
        char error_str[error_len];
        sprintf(error_str, "ERR Invalid DAG command. Unexpected argument:  %s", arg_string);
        RAI_SetError(rinfo->err, RAI_EDAGBUILDER, error_str);
        return REDISMODULE_ERR;
    }
    // This verification is needed for AI.DAGEXECUTE(_RO) commands (but not for the deprecated DAG
    // commands).
    if (!strncasecmp(RedisModule_StringPtrLen(argv[0], NULL), "AI.DAGEXECUTE",
                     strlen("AI.DAGEXECUTE"))) {
        if (!load_complete && !persist_complete && !routing_complete) {
            RAI_SetError(rinfo->err, RAI_EDAGBUILDER,
                         "ERR AI.DAGEXECUTE and AI.DAGEXECUTE_RO commands must "
                         "contain at least one out of ROUTING, LOAD, PERSIST keywords");
            return REDISMODULE_ERR;
        }
    }
    if (array_len(*dag_ops) < 1) {
        RAI_SetError(rinfo->err, RAI_EDAGBUILDER, "ERR DAG is empty");
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

int ParseDAGExecuteCommand(RedisAI_RunInfo *rinfo, RedisModuleCtx *ctx, RedisModuleString **argv,
                           int argc, bool dag_ro) {

    // The minimal command is of the form: AI.DAGEXECUTE(_RO) ROUTING/LOAD/PERSIST 1 <key> |>
    // AI.TENSORGET <key>
    if (argc < 6) {
        if (dag_ro) {
            RAI_SetError(rinfo->err, RAI_EDAGBUILDER,
                         "ERR missing arguments for 'AI.DAGEXECUTE_RO' command");
        } else {
            RAI_SetError(rinfo->err, RAI_EDAGBUILDER,
                         "ERR missing arguments for 'AI.DAGEXECUTE' command");
        }
        return REDISMODULE_ERR;
    }

    // First we parse KEYS, LOAD, PERSIST and TIMEOUT parts, and we collect the DAG ops' args.
    array_new_on_stack(RAI_DagOp *, 10, dag_ops);
    if (DAGInitialParsing(rinfo, ctx, argv, argc, dag_ro, &dag_ops) != REDISMODULE_OK) {
        goto cleanup;
    }

    if (ParseDAGExecuteOps(rinfo, dag_ops, dag_ro) != REDISMODULE_OK) {
        goto cleanup;
    }

    if (MapTensorsKeysToIndices(rinfo, rinfo->tensorsNamesToIndices) != REDISMODULE_OK) {
        goto cleanup;
    }
    if (ValidatePersistKeys(rinfo, rinfo->tensorsNamesToIndices, rinfo->persistTensors) !=
        REDISMODULE_OK) {
        goto cleanup;
    }
    AI_dictRelease(rinfo->tensorsNamesToIndices);
    rinfo->tensorsNamesToIndices = NULL;
    array_free(dag_ops);
    return REDISMODULE_OK;

cleanup:
    for (size_t i = 0; i < array_len(dag_ops); i++) {
        RAI_FreeDagOp(dag_ops[i]);
    }
    // For the case that error was raised after the ops were inserted to the run info.
    array_clear(rinfo->dagOps);
    array_free(dag_ops);
    return REDISMODULE_ERR;
}
