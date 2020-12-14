
#include "dag_llapi.h"
#include "dag.h"
#include "redisai.h"
#include "run_info.h"
#include "modelRun_ctx.h"


RAI_DAGRunCtx *RAI_DagFromString(const char* dag_str, RAI_Error* err) {

	// Put here the DAG from string function (maybe change the RunInfo).

}

int RAI_DagRunAsync(RedisAI_RunInfo *run_info, RAI_OnFinishCB DAGAsyncFinish,
  void *private_data) {

	run_info->OnFinish = (RedisAI_OnFinishCB) DAGAsyncFinish;
	run_info->private_data = private_data;
	return(DAG_InsertDAGToQueue(run_info) == REDISMODULE_ERR);
}

RAI_DAGRunCtx *RAI_DagRunCtxCreate() {

	RedisAI_RunInfo *rinfo = NULL;
	if (RAI_InitRunInfo(&rinfo) == REDISMODULE_ERR)
		return NULL;
	return (RAI_DAGRunCtx *)rinfo;
}
/*
int RAI_DagAddTensorSet(RAI_DAGRunCtx *run_info, const char* t_name,
  RAI_Tensor *t, RAI_Error* err) {
	RedisAI_RunInfo *rinfo = (RedisAI_RunInfo *)run_info;

	RAI_DagOp *currentDagOp = NULL;
	RAI_InitDagOp(&currentDagOp);
	rinfo->dagOps = array_append(rinfo->dagOps, currentDagOp);
	rinfo->dagOpCount++;

	rinfo->dagOps[rinfo->dagOpCount]->commandType = REDISAI_DAG_CMD_TENSORSET;
	rinfo->dagOps[rinfo->dagOpCount]->devicestr = "CPU";

}
*/
int RAI_ModelRunAsync(RAI_ModelRunCtx* mctxs, RAI_OnFinishCB ModelAsyncFinish,
  void *private_data) {

	RAI_Error error = {0};
	RedisAI_RunInfo *rinfo = Dag_CreateFromSingleModelRunOp(mctxs, &error,
	  NULL, NULL, NULL, 0);
	if (rinfo == NULL)
		// Call OnFinish with the error
		return REDISMODULE_ERR;
	rinfo->OnFinish = (RedisAI_OnFinishCB)ModelAsyncFinish;
	rinfo->private_data = private_data;
	return DAG_InsertDAGToQueue(rinfo);
	//should update inkeys and outkeys?
}
