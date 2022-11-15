/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "redismodule.h"
#include "execution/run_info.h"

/**
 @brief We are given a DAG runInfo of a sequence of operations, each with its own
 input and output keys. The names of the keys will be used to look whether the
 inputs to a DAG operation have all been realized by previous operations (or if
 they are available as part of LOADed keys from keyspace).
 This strategy is fine if keys are not aliased, that is, if a command's output
 overwrites the key of a previous command. This would trick DAG operations into
 thinking that their input is ready when it's not.
 To overcome this, we map the input and output tensors for every operation to indices,
 in the following way. For every input of an operation having the key "x", we map the index
 for which "x" was last mapped to when, it was an output of a previous operation.
 For every output of an operation "y", we map the next available index in the array.
 Every entry in the DAG array contains NULL (except for tensors that where loaded
 before the DAG run starts).
 @param rinfo The DAG runInfo.
 @param tensorsNamesToInd A dict mapping every key name of a tensor that appeared
 in DAG operation, to the maximal index of the DAG shared array for which they were mapped to.
 @returns REDISMODULE_ERR if there exists an operation for which one of the input
 tensors didn't appear as an output of a previous operation, REDISMODULE_OK otherwise
 */
int MapTensorsKeysToIndices(RedisAI_RunInfo *rinfo, AI_dict *tensorsNamesToInd);

/**
 * @brief Validates that tensors key names to persist appeared in the DAG operations.
 * @param rinfo The DAG runInfo.
 * @param tensorsNamesToInd A dict mapping every key name of a tensor that appeared
 * in DAG operation, to the maximal index of the DAG shared array for which they were mapped to.
 * @param persistTensorsNames A hash table the contains the names of the tensors
 * to persist when the DAG run is finished.
 * @return REDISMODULE_ERR if there exists a tensor key to persist that didn't
 * appear in DAG operation, REDISMODULE_OK otherwise
 */
int ValidatePersistKeys(RedisAI_RunInfo *rinfo, AI_dict *tensorsNamesToInd,
                        AI_dict *persistTensorsNames);

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
const RAI_Tensor *RAI_DAGOutputTensor(RAI_OnFinishCtx *finish_ctx, size_t index);

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
const RAI_Error *RAI_DAGGetError(RAI_OnFinishCtx *finish_ctx);
