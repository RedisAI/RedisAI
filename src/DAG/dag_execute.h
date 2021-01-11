#pragma once

#include "redismodule.h"
#include "run_info.h"

/**
 * @brief We are given a DAG runInfo of a sequence of operations, each with its own
 input and output keys. The names of the keys will be used to look whether the
 inputs to a DAG operation have all been realized by previous operations (or if
 they are available as part of LOADed keys from keyspace).
 This strategy is fine if keys are not aliased, that is, if a command's output
 overwrites the key of a previous command. This would trick DAG operations into
 thinking that their input is ready when it's not.
 To overcome this, we make key names unique, so that names are not aliased. We
 mangle the names by appending a numerical suffix ":0001". After computing, we
 demangle the keys in order to persist them.*/
int MangleTensorsNames(RedisAI_RunInfo *rinfo);

int RAI_DAGRun(RAI_DAGRunCtx *run_info, RAI_OnFinishCB DAGAsyncFinish, void *private_data,
               RAI_Error *err);

size_t RAI_DAGNumOutputs(RAI_OnFinishCtx *finish_ctx);

RAI_Tensor *RAI_DAGOutputTensor(RAI_OnFinishCtx *finish_ctx, size_t index);

int RAI_DAGRunError(RAI_OnFinishCtx *finish_ctx);

RAI_Error *RAI_DAGCopyOpStatus(RAI_OnFinishCtx *finish_ctx, size_t index);
