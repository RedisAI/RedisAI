
#include "command_parser.h"
#include "redismodule.h"
#include "run_info.h"
#include "modelRun_ctx.h"
#include "DAG/dag.h"
#include "DAG/dag_parser.h"

static int parseTimeout(RedisModuleString *timeout_arg, RAI_Error *error, long long *timeout) {

    const int retval = RedisModule_StringToLongLong(timeout_arg, timeout);
    if (retval != REDISMODULE_OK || timeout <= 0) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR Invalid value for TIMEOUT");
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

int ParseModelRunCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc, RAI_Model **model,
                         RAI_Error *error, RedisModuleString ***inkeys,
                         RedisModuleString ***outkeys, RedisModuleString **runkey,
                         long long *timeout) {

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
    RedisModule_RetainString(NULL, argv[argpos]);
    *runkey = argv[argpos];
    const char *arg_string = RedisModule_StringPtrLen(argv[++argpos], NULL);

    // Parse timeout arg if given and store it in timeout
    if (!strcasecmp(arg_string, "TIMEOUT")) {
        if (parseTimeout(argv[++argpos], error, timeout) == REDISMODULE_ERR)
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
            RedisModule_RetainString(NULL, argv[argpos]);
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

int ProcessModelRunCmd(RedisAI_RunInfo *rinfo, RedisModuleCtx *ctx, RedisModuleString **argv,
                       int argc) {

    // Build a ModelRunCtx from command.
    RAI_Error error = {0};
    RAI_Model *model;
    RedisModuleString **inkeys = array_new(RedisModuleString *, 1);
    RedisModuleString **outkys = array_new(RedisModuleString *, 1);
    RedisModuleString *runkey;
    long long timeout = 0;
    if (ParseModelRunCommand(ctx, argv, argc, &model, &error, &inkeys, &outkys, &runkey,
                             &timeout) == REDISMODULE_ERR) {
        RedisModule_ReplyWithError(ctx, RAI_GetErrorOneLine(&error));
        return REDISMODULE_ERR;
    }
    RAI_ModelRunCtx *mctx = RAI_ModelRunCtxCreate(model);

    if (rinfo->single_op_dag) {
        // Set params in ModelRunCtx, bring inputs from key space.
        if (ModelRunCtx_SetParams(ctx, inkeys, outkys, mctx) == REDISMODULE_ERR)
            return REDISMODULE_ERR;
    }

    if (Dag_PopulateSingleModelRunOp(rinfo, mctx, inkeys, outkys, runkey, timeout) ==
        REDISMODULE_ERR) {
        RedisModule_ReplyWithError(ctx, RAI_GetErrorOneLine(rinfo->err));
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

int ProcessRunCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc, int command,
                      int dagMode) {

    int flags = RedisModule_GetContextFlags(ctx);
    bool blocking_not_allowed = (flags & (REDISMODULE_CTX_FLAGS_MULTI | REDISMODULE_CTX_FLAGS_LUA));
    if (blocking_not_allowed)
        return RedisModule_ReplyWithError(
            ctx, "ERR Cannot run RedisAI command within a transaction or a LUA script");

    RedisAI_RunInfo *rinfo = NULL;
    if (RAI_InitRunInfo(&rinfo) == REDISMODULE_ERR) {
        RedisModule_ReplyWithError(
            ctx, "ERR Unable to allocate the memory and initialise the RedisAI_RunInfo structure");
        return REDISMODULE_ERR;
    }
    int status = REDISMODULE_ERR;
    switch (command) {
    case CMD_MODELRUN:
        rinfo->single_op_dag = 1;
        status = ProcessModelRunCmd(rinfo, ctx, argv, argc);
        break;
    case CMD_SCRIPTRUN:
        rinfo->single_op_dag = 1;
        status = DAG_CommandParser(ctx, argv, argc, REDISAI_DAG_WRITE_MODE, &rinfo);
        break;
    case CMD_DAGRUN:
        status = DAG_CommandParser(ctx, argv, argc, dagMode, &rinfo);
        break;
    default:
        status = REDISMODULE_ERR;
    }
    if (status == REDISMODULE_ERR)
        return REDISMODULE_OK;

    // Block the client before adding rinfo to the run queues (sync call).
    rinfo->client = RedisModule_BlockClient(ctx, RedisAI_DagRun_Reply, NULL, RunInfo_FreeData, 0);
    RedisModule_SetDisconnectCallback(rinfo->client, RedisAI_Disconnected);
    rinfo->OnFinish = DAG_ReplyAndUnblock;
    return DAG_InsertDAGToQueue(rinfo);
}
