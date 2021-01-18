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

/**
 * @brief Run asynchronously a DAG. This will validate that the sequence of DAG ops
 * is valid and generate a unique key to the tensor that flow in the DAG (mangleTensorsNames)
 * Then, DAG is sent to the devices' run queues and will be execute by a workung thread.
 * @param DAGAsyncFinish This is a callback that will be called after the whole DAG finish its run.
 * @param private_data This is an input to the DAGAsyncFinish callback. Can be used to save the
 * results and errors
 * @param err Error is returned in case that the validation failed, and the DAG wasn't inserted to
 * the queues.
 */
int RAI_DAGRun(RAI_DAGRunCtx *run_info, RAI_OnFinishCB DAGAsyncFinish, void *private_data,
               RAI_Error *err);

/**
 * @brief This can be called in the finish CB, returns the number of outputs (TENSORGET ops).
 * @param finish_ctx This represents the DAG runInfo at the end of the run.
 */
size_t RAI_DAGNumOutputs(RAI_OnFinishCtx *finish_ctx);

/**
 * @brief This can be called in the finish CB, returns a specific output tensor (result of a
 * TENSORGET op).
 * @param finish_ctx This represents the DAG runInfo at the end of the run.
 * @param index The index of the TENSORGET op in the DAG.
 * @retval returns the tensor that the i'th TENSORGET op outputs.
 */
RAI_Tensor *RAI_DAGOutputTensor(RAI_OnFinishCtx *finish_ctx, size_t index);

/**
 * @brief Returns true if (at least) one of the DAG ops encountered an error.
 */
bool RAI_DAGRunError(RAI_OnFinishCtx *finish_ctx);

/**
 * @brief This can be called in the finish CB, to get DAG error details.
 * @param finish_ctx This represents the DAG runInfo at the end of the run.
 * @retval returns an object that represents the DAG status, from which a user can
 * obtain the error code (error code is "OK" if no error has occurred) and error details.
 */
RAI_Error *RAI_DAGGetError(RAI_OnFinishCtx *finish_ctx);
