#ifndef REDISAI_ASYNC_LLAPI_H
#define REDISAI_ASYNC_LLAPI_H

#include "model_struct.h"
#include "run_info.h"

/**
 * Insert the ModelRunCtx to the run queues so it will run asynchronously.
 *
 * @param mctx ModelRunCtx to execute
 * @param ModelAsyncFinish A callback that will be called when the execution is finished.
 * @param private_data This is going to be sent to to the ModelAsyncFinish.
 * @return REDISMODULE_OK if the mctx was insert to the queues successfully, REDISMODULE_ERR
 * otherwise.
 */

int RAI_ModelRunAsync(RAI_ModelRunCtx *mctx, RAI_OnFinishCB ModelAsyncFinish, void *private_data);

#endif // REDISAI_ASYNC_LLAPI_H
