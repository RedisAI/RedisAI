#include "dag.h"
#include "redisai.h"
#include "run_info.h"
#include "async_llapi.h"
#include "modelRun_ctx.h"

/* This file contains the implementation of the RedisAI low level API
 * for running models, scripts and DAGs asynchronously.
 */

int RAI_ModelRunAsync(RAI_ModelRunCtx *mctx, RAI_OnFinishCB ModelAsyncFinish, void *private_data) {

    RAI_Error error = {0};
    RedisAI_RunInfo *rinfo = Dag_CreateFromSingleModelRunOp(mctx, &error, NULL, NULL, NULL, 0);
    if (rinfo == NULL)
        return REDISMODULE_ERR;
    rinfo->OnFinish = (RedisAI_OnFinishCB)ModelAsyncFinish;
    rinfo->private_data = private_data;
    return DAG_InsertDAGToQueue(rinfo);
}
