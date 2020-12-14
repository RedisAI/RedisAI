#ifndef REDISAI_DAG_LLAPI_H
#define REDISAI_DAG_LLAPI_H

#include "model_struct.h"
#include "run_info.h"

int RAI_ModelRunAsync(RAI_ModelRunCtx* mctxs, RAI_OnFinishCB ModelAsyncFinish,
  void *private_data);
#endif //REDISAI_DAG_LLAPI_H
