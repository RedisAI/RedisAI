#include "model_commands_parser.h"
#include "redis_ai_objects/model.h"
#include "util/string_utils.h"
#include "execution/parsing/parse_utils.h"
#include "execution/execution_contexts/modelRun_ctx.h"

static int _ModelExecuteCommand_ParseArgs(RedisModuleCtx *ctx, int argc, RedisModuleString **argv,
                                          RAI_Model **model, RAI_Error *error,
                                          RedisModuleString ***inkeys, RedisModuleString ***outkeys,
                                          RedisModuleString **runkey, long long *timeout) {

    if (argc < 8) {
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR wrong number of arguments for 'AI.MODELEXECUTE' command");
        return REDISMODULE_ERR;
    }
    size_t arg_pos = 1;
    const int status = RAI_GetModelFromKeyspace(ctx, argv[arg_pos], model, REDISMODULE_READ, error);
    if (status == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }
    *runkey = RAI_HoldString(NULL, argv[arg_pos++]);
    const char *arg_string = RedisModule_StringPtrLen(argv[arg_pos++], NULL);

    if (strcasecmp(arg_string, "INPUTS") != 0) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR INPUTS not specified");
        return REDISMODULE_ERR;
    }

    long long ninputs = 0, noutputs = 0;
    if (RedisModule_StringToLongLong(argv[arg_pos++], &ninputs) != REDISMODULE_OK) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR Invalid argument for input_count");
        return REDISMODULE_ERR;
    }
    if (ninputs <= 0) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR Input count must be a positive integer");
        return REDISMODULE_ERR;
    }
    if ((*model)->ninputs != ninputs) {
        RAI_SetError(error, RAI_EMODELRUN,
                     "Number of keys given as INPUTS here does not match model definition");
        return REDISMODULE_ERR;
    }
    // arg_pos = 4
    size_t first_input_pos = arg_pos;
    if (first_input_pos + ninputs > argc) {
        RAI_SetError(
            error, RAI_EMODELRUN,
            "ERR number of input keys to AI.MODELEXECUTE command does not match the number of "
            "given arguments");
        return REDISMODULE_ERR;
    }
    for (; arg_pos < first_input_pos + ninputs; arg_pos++) {
        *inkeys = array_append(*inkeys, RAI_HoldString(NULL, argv[arg_pos]));
    }

    if (argc == arg_pos ||
        strcasecmp(RedisModule_StringPtrLen(argv[arg_pos++], NULL), "OUTPUTS") != 0) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR OUTPUTS not specified");
        return REDISMODULE_ERR;
    }
    if (argc == arg_pos ||
        RedisModule_StringToLongLong(argv[arg_pos++], &noutputs) != REDISMODULE_OK) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR Invalid argument for output_count");
    }
    if (noutputs <= 0) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR Output count must be a positive integer");
        return REDISMODULE_ERR;
    }
    if ((*model)->noutputs != noutputs) {
        RAI_SetError(error, RAI_EMODELRUN,
                     "Number of keys given as OUTPUTS here does not match model definition");
        return REDISMODULE_ERR;
    }
    // arg_pos = ninputs+6, the argument that we already parsed are:
    // AI.MODELEXECUTE <model_key> INPUTS <input_count> <input> ... OUTPUTS <output_count>
    size_t first_output_pos = arg_pos;
    if (first_output_pos + noutputs > argc) {
        RAI_SetError(
            error, RAI_EMODELRUN,
            "ERR number of output keys to AI.MODELEXECUTE command does not match the number of "
            "given arguments");
        return REDISMODULE_ERR;
    }
    for (; arg_pos < first_output_pos + noutputs; arg_pos++) {
        *outkeys = array_append(*outkeys, RAI_HoldString(NULL, argv[arg_pos]));
    }
    if (arg_pos == argc) {
        return REDISMODULE_OK;
    }

    // Parse timeout arg if given and store it in timeout.
    char *error_str;
    arg_string = RedisModule_StringPtrLen(argv[arg_pos++], NULL);
    if (!strcasecmp(arg_string, "TIMEOUT")) {
        if (arg_pos == argc) {
            RAI_SetError(error, RAI_EMODELRUN, "ERR No value provided for TIMEOUT");
            return REDISMODULE_ERR;
        }
        if (ParseTimeout(argv[arg_pos++], error, timeout) == REDISMODULE_ERR)
            return REDISMODULE_ERR;
    } else {
        error_str = RedisModule_Alloc(strlen("Invalid argument: ") + strlen(arg_string) + 1);
        sprintf(error_str, "Invalid argument: %s", arg_string);
        RAI_SetError(error, RAI_EMODELRUN, error_str);
        RedisModule_Free(error_str);
        return REDISMODULE_ERR;
    }

    // There are no more valid args to be processed.
    if (arg_pos != argc) {
        arg_string = RedisModule_StringPtrLen(argv[arg_pos], NULL);
        error_str = RedisModule_Alloc(strlen("Invalid argument: ") + strlen(arg_string) + 1);
        sprintf(error_str, "Invalid argument: %s", arg_string);
        RAI_SetError(error, RAI_EMODELRUN, error_str);
        RedisModule_Free(error_str);
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

int ParseModelExecuteCommand(RedisAI_RunInfo *rinfo, RAI_DagOp *currentOp, RedisModuleString **argv,
                             int argc) {

    int res = REDISMODULE_ERR;
    // Build a ModelRunCtx from command.
    RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(NULL);
    RAI_Model *model;
    long long timeout = 0;
    if (_ModelExecuteCommand_ParseArgs(ctx, argc, argv, &model, rinfo->err, &currentOp->inkeys,
                                       &currentOp->outkeys, &currentOp->runkey,
                                       &timeout) == REDISMODULE_ERR) {
        goto cleanup;
    }

    if (timeout > 0 && !rinfo->single_op_dag) {
        RAI_SetError(rinfo->err, RAI_EDAGBUILDER, "ERR TIMEOUT not allowed within a DAG command");
        goto cleanup;
    }

    RAI_ModelRunCtx *mctx = RAI_ModelRunCtxCreate(model);
    currentOp->commandType = REDISAI_DAG_CMD_MODELRUN;
    currentOp->mctx = mctx;
    currentOp->devicestr = mctx->model->devicestr;

    if (rinfo->single_op_dag) {
        rinfo->timeout = timeout;
        // Set params in ModelRunCtx, bring inputs from key space.
        if (ModelRunCtx_SetParams(ctx, currentOp->inkeys, currentOp->outkeys, mctx, rinfo->err) ==
            REDISMODULE_ERR)
            goto cleanup;
    }
    res = REDISMODULE_OK;

cleanup:
    RedisModule_FreeThreadSafeContext(ctx);
    return res;
}
