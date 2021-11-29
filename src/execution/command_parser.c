#include "command_parser.h"
#include "redismodule.h"
#include "execution/run_info.h"
#include "execution/DAG/dag.h"
#include "execution/parsing/dag_parser.h"
#include "execution/parsing/deprecated.h"
#include "execution/parsing/model_commands_parser.h"
#include "execution/parsing/script_commands_parser.h"

int RedisAI_ExecuteCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc,
                           RunCommand command, bool ro_dag) {

    int flags = RedisModule_GetContextFlags(ctx);
    bool blocking_not_allowed = (flags & (REDISMODULE_CTX_FLAGS_MULTI | REDISMODULE_CTX_FLAGS_LUA));
    if (blocking_not_allowed)
        return RedisModule_ReplyWithError(
            ctx, "ERR Cannot run RedisAI command within a transaction or a LUA script");

    RedisAI_RunInfo *rinfo;
    RAI_InitRunInfo(&rinfo);
    int status = REDISMODULE_ERR;

    switch (command) {
    case CMD_MODELRUN:
        rinfo->single_op_dag = 1;
        RAI_DagOp *modelRunOp;
        RAI_InitDagOp(&modelRunOp);
        rinfo->dagOps = array_append(rinfo->dagOps, modelRunOp);
        status = ParseModelRunCommand(rinfo, modelRunOp, argv, argc);
        break;
    case CMD_SCRIPTRUN:
        rinfo->single_op_dag = 1;
        RAI_DagOp *scriptRunOp;
        RAI_InitDagOp(&scriptRunOp);
        rinfo->dagOps = array_append(rinfo->dagOps, scriptRunOp);
        status = ParseScriptRunCommand(rinfo, scriptRunOp, argv, argc);
        break;
    case CMD_DAGRUN:
        status = ParseDAGRunCommand(rinfo, ctx, argv, argc, ro_dag);
        break;
    case CMD_MODELEXECUTE:
        rinfo->single_op_dag = 1;
        RAI_DagOp *modelExecuteOp;
        RAI_InitDagOp(&modelExecuteOp);
        rinfo->dagOps = array_append(rinfo->dagOps, modelExecuteOp);
        status = ParseModelExecuteCommand(rinfo, modelExecuteOp, argv, argc);
        break;
    case CMD_SCRIPTEXECUTE:
        rinfo->single_op_dag = 1;
        RAI_DagOp *scriptExecOp;
        RAI_InitDagOp(&scriptExecOp);
        rinfo->dagOps = array_append(rinfo->dagOps, scriptExecOp);
        status = ParseScriptExecuteCommand(rinfo, scriptExecOp, argv, argc);
        break;
    case CMD_DAGEXECUTE:
        status = ParseDAGExecuteCommand(rinfo, ctx, argv, argc, ro_dag);
        break;
    default:
        break;
    }
    if (status == REDISMODULE_ERR) {
        RedisModule_ReplyWithError(ctx, RAI_GetErrorOneLine(rinfo->err));
        RAI_FreeRunInfo(rinfo);
        return REDISMODULE_OK;
    }
    rinfo->dagOpCount = array_len(rinfo->dagOps);

    rinfo->OnFinish = DAG_ReplyAndUnblock;
    rinfo->client = RedisModule_BlockClient(ctx, RedisAI_DagRun_Reply, NULL, RunInfo_FreeData, 0);
    if (DAG_InsertDAGToQueue(rinfo) != REDISMODULE_OK) {
        RedisModule_UnblockClient(rinfo->client, rinfo);
        return REDISMODULE_ERR;
    }
    if (RedisModule_GetServerVersion() >= 0x060200) { // The following command is supported only from redis 6.2
        RedisModule_BlockedClientMeasureTimeStart(rinfo->client);
    }
    return REDISMODULE_OK;
}
