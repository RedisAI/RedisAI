
#include "command_parser.h"
#include "redismodule.h"
#include "run_info.h"
#include "modelRun_ctx.h"
#include "DAG/dag.h"
#include "DAG/dag_parser.h"
#include "util/string_utils.h"

static int _parseTimeout(RedisModuleString *timeout_arg, RAI_Error *error, long long *timeout) {

    const int retval = RedisModule_StringToLongLong(timeout_arg, timeout);
    if (retval != REDISMODULE_OK || timeout <= 0) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR Invalid value for TIMEOUT");
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

static int _ModelRunCommand_ParseArgs(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                                      RAI_Model **model, RAI_Error *error,
                                      RedisModuleString ***inkeys, RedisModuleString ***outkeys,
                                      RedisModuleString **runkey, long long *timeout) {

    if (argc < 4) {
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR wrong number of arguments for 'AI.MODELRUN' command");
        return REDISMODULE_ERR;
    }
    size_t argpos = 1;
    RedisModuleKey *modelKey;
    const int status =
        RAI_GetModelFromKeyspace(ctx, argv[argpos], &modelKey, model, REDISMODULE_READ);
    if (status == REDISMODULE_ERR) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR Model not found");
        return REDISMODULE_ERR;
    }
    RAI_HoldString(NULL, argv[argpos]);
    *runkey = argv[argpos];
    const char *arg_string = RedisModule_StringPtrLen(argv[++argpos], NULL);

    // Parse timeout arg if given and store it in timeout
    if (!strcasecmp(arg_string, "TIMEOUT")) {
        if (_parseTimeout(argv[++argpos], error, timeout) == REDISMODULE_ERR)
            return REDISMODULE_ERR;
        arg_string = RedisModule_StringPtrLen(argv[++argpos], NULL);
    }
    if (strcasecmp(arg_string, "INPUTS") != 0) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR INPUTS not specified");
        return REDISMODULE_ERR;
    }

    bool is_input = true, is_output = false;
    size_t ninputs = 0, noutputs = 0;

    while (++argpos < argc) {
        arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
        if (!strcasecmp(arg_string, "OUTPUTS") && !is_output) {
            is_input = false;
            is_output = true;
        } else {
            RAI_HoldString(NULL, argv[argpos]);
            if (is_input) {
                ninputs++;
                *inkeys = array_append(*inkeys, argv[argpos]);
            } else {
                noutputs++;
                *outkeys = array_append(*outkeys, argv[argpos]);
            }
        }
    }
    if ((*model)->inputs && (*model)->ninputs != ninputs) {
        RAI_SetError(error, RAI_EMODELRUN,
                     "Number of names given as INPUTS during MODELSET and keys given as "
                     "INPUTS here do not match");
        return REDISMODULE_ERR;
    }

    if ((*model)->outputs && (*model)->noutputs != noutputs) {
        RAI_SetError(error, RAI_EMODELRUN,
                     "Number of names given as OUTPUTS during MODELSET and keys given as "
                     "OUTPUTS here do not match");
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

/**
 * Extract the params for the ModelCtxRun object from AI.MODELRUN arguments.
 *
 * @param ctx Context in which Redis modules operate
 * @param inkeys Model input tensors keys, as an array of strings
 * @param outkeys Model output tensors keys, as an array of strings
 * @param mctx Destination Model context to store the parsed data
 * @return REDISMODULE_OK in case of success, REDISMODULE_ERR otherwise
 */

static int _ModelRunCtx_SetParams(RedisModuleCtx *ctx, RedisModuleString **inkeys,
                                  RedisModuleString **outkeys, RAI_ModelRunCtx *mctx) {

    RAI_Model *model = mctx->model;
    RAI_Tensor *t;
    RedisModuleKey *key;
    char *opname = NULL;
    size_t ninputs = array_len(inkeys), noutputs = array_len(outkeys);
    for (size_t i = 0; i < ninputs; i++) {
        const int status = RAI_GetTensorFromKeyspace(ctx, inkeys[i], &key, &t, REDISMODULE_READ);
        if (status == REDISMODULE_ERR) {
            RedisModule_Log(ctx, "warning", "could not load tensor %s from keyspace",
                            RedisModule_StringPtrLen(inkeys[i], NULL));
            return REDISMODULE_ERR;
        }
        if (model->inputs)
            opname = model->inputs[i];
        RAI_ModelRunCtxAddInput(mctx, opname, t);
    }
    for (size_t i = 0; i < noutputs; i++) {
        if (model->outputs)
            opname = model->outputs[i];
        RAI_ModelRunCtxAddOutput(mctx, opname);
    }
    return REDISMODULE_OK;
}

int ParseModelRunCommand(RedisAI_RunInfo *rinfo, RedisModuleCtx *ctx, RedisModuleString **argv,
                         int argc) {

    // Build a ModelRunCtx from command.
    RAI_Error error = {0};
    RAI_Model *model;
    RedisModuleString **inkeys = array_new(RedisModuleString *, 1);
    RedisModuleString **outkeys = array_new(RedisModuleString *, 1);
    RedisModuleString *runkey = NULL;
    RAI_ModelRunCtx *mctx = NULL;
    RAI_DagOp *currentOp;

    long long timeout = 0;
    if (_ModelRunCommand_ParseArgs(ctx, argv, argc, &model, &error, &inkeys, &outkeys, &runkey,
                                   &timeout) == REDISMODULE_ERR) {
        RedisModule_ReplyWithError(ctx, RAI_GetErrorOneLine(&error));
        goto cleanup;
    }
    mctx = RAI_ModelRunCtxCreate(model);

    if (rinfo->single_op_dag) {
        rinfo->timeout = timeout;
        // Set params in ModelRunCtx, bring inputs from key space.
        if (_ModelRunCtx_SetParams(ctx, inkeys, outkeys, mctx) == REDISMODULE_ERR)
            goto cleanup;
    }
    if (RAI_InitDagOp(&currentOp) == REDISMODULE_ERR) {
        RedisModule_ReplyWithError(
            ctx, "ERR Unable to allocate the memory and initialise the RAI_dagOp structure");
        goto cleanup;
    }
    currentOp->commandType = REDISAI_DAG_CMD_MODELRUN;
    Dag_PopulateOp(currentOp, mctx, inkeys, outkeys, runkey);
    rinfo->dagOps = array_append(rinfo->dagOps, currentOp);
    return REDISMODULE_OK;

cleanup:
    for (size_t i = 0; i < array_len(inkeys); i++) {
        RedisModule_FreeString(NULL, inkeys[i]);
    }
    array_free(inkeys);
    for (size_t i = 0; i < array_len(outkeys); i++) {
        RedisModule_FreeString(NULL, outkeys[i]);
    }
    array_free(outkeys);
    if (runkey)
        RedisModule_FreeString(NULL, runkey);
    if (mctx)
        RAI_ModelRunCtxFree(mctx);
    RAI_FreeRunInfo(rinfo);
    return REDISMODULE_ERR;
}

int RedisAI_ExecuteCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                           RunCommand command, bool ro_dag) {

    int flags = RedisModule_GetContextFlags(ctx);
    bool blocking_not_allowed = (flags & (REDISMODULE_CTX_FLAGS_MULTI | REDISMODULE_CTX_FLAGS_LUA));
    if (blocking_not_allowed)
        return RedisModule_ReplyWithError(
            ctx, "ERR Cannot run RedisAI command within a transaction or a LUA script");

    RedisAI_RunInfo *rinfo = NULL;
    if (RAI_InitRunInfo(&rinfo) != REDISMODULE_OK) {
        RedisModule_ReplyWithError(
            ctx, "ERR Unable to allocate the memory and initialize the RedisAI_RunInfo structure");
        return REDISMODULE_ERR;
    }

    int status = REDISMODULE_ERR;
    switch (command) {
    case CMD_MODELRUN:
        rinfo->single_op_dag = 1;
        status = ParseModelRunCommand(rinfo, ctx, argv, argc);
        break;
    case CMD_SCRIPTRUN:
        rinfo->single_op_dag = 1;
        status = DAG_CommandParser(ctx, argv, argc, ro_dag, &rinfo);
        break;
    case CMD_DAGRUN:
        status = DAG_CommandParser(ctx, argv, argc, ro_dag, &rinfo);
        break;
    default:
        status = REDISMODULE_ERR;
    }
    if (status == REDISMODULE_ERR) {
        return REDISMODULE_OK;
    }

    rinfo->dagOpCount = array_len(rinfo->dagOps);

    // Block the client before adding rinfo to the run queues (sync call).
    rinfo->client = RedisModule_BlockClient(ctx, RedisAI_DagRun_Reply, NULL, RunInfo_FreeData, 0);
    RedisModule_SetDisconnectCallback(rinfo->client, RedisAI_Disconnected);
    rinfo->OnFinish = DAG_ReplyAndUnblock;
    return DAG_InsertDAGToQueue(rinfo);
}
