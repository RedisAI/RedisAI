#pragma once

#include "redisai.h"

/**
 * @brief Create a new empty DAG runInfo object.
 */
RAI_DAGRunCtx *RAI_DAGRunCtxCreate(void);

/**
 * @brief Create a new MODELRUN op for a DAG.
 * @param model The model to run.
 */
RAI_DAGRunOp *RAI_DAGCreateModelRunOp(RAI_Model *model);

/**
 * @brief Create a new SCRIPTRUN op for a DAG.
 * @param script The script to run.
 * @param func_name The specific function to run in the given script.
 */
RAI_DAGRunOp *RAI_DAGCreateScriptRunOp(RAI_Script *script, const char *func_name);

/**
 * @brief Add an input key to a DAG run op (before inserting it to the DAG).
 * @param DAGop The DAG run op (MODELRUN / SCRIPTRUN).
 * @param input The tensor input name (this name should appear in a previous op of the DAG).
 */
int RAI_DAGRunOpAddInput(RAI_DAGRunOp *DAGOp, const char *input);

/**
 * @brief Add an output key to a DAG run op (before inserting it to the DAG).
 * @param DAGop The DAG run op (MODELRUN / SCRIPTRUN).
 * @param output The tensor output name (this name may appear in one of the following ops of the
 * DAG).
 */
int RAI_DAGRunOpAddOutput(RAI_DAGRunOp *DAGOp, const char *output);

/**
 * @brief Add a run op (MODELRUN/SCRIPTRUN) to a DAG.
 * @param runInfo The DAG to insert the op to.
 * @param DAGop The DAG run op (MODELRUN / SCRIPTRUN).
 * @param err Error is returned in case of a MODELRUN op if the number of inputs and outputs
 * given to the op does not match to the number of inputs and outputs in the model definition.
 */
int RAI_DAGAddRunOp(RAI_DAGRunCtx *run_info, RAI_DAGRunOp *DAGop, RAI_Error *err);

/**
 * @brief Load a tensor from keyspace to the DAG local context.
 * @param runInfo The DAG to load the tensor into.
 * @param tname The tensor key.
 * @param err Error is returned in case that the key does not exist, or not holding a tensor type.
 */
int RAI_DAGLoadTensor(RAI_DAGRunCtx *run_info, const char *t_name, RAI_Error *err);

/**
 * @brief Load a tensor from keyspace to the DAG local context.
 * @param runInfo The DAG to load the tensor into.
 * @param tname The tensor key (can hold any binary string).
 * @param err Error is returned in case that the key does not exist, or not holding a tensor type.
 */
int RAI_DAGLoadTensorRS(RAI_DAGRunCtx *run_info, RedisModuleString *t_name, RAI_Error *err);

/**
 * @brief Append a TENSORSET op to a DAG (can use to load an intermediate tensors)
 * @param runInfo The DAG to append this op into.
 * @param tensor The tensor to set.
 */
int RAI_DAGAddTensorSet(RAI_DAGRunCtx *run_info, const char *t_name, RAI_Tensor *tensor);

/**
 * @brief Append a TENSORGET op to a DAG (can use to output intermediate and final tensors)
 * @param runInfo The DAG to append this op into.
 * @param tensor The tensor to set.
 */
int RAI_DAGAddTensorGet(RAI_DAGRunCtx *run_info, const char *t_name, RAI_Error *err);

/**
 * @brief Returns the number of ops in a DAG.
 */
size_t RAI_DAGNumOps(RAI_DAGRunCtx *run_info);

/**
 * @brief Free DAG's runInfo and all its internal ops.
 */
void RAI_DAGFree(RAI_DAGRunCtx *run_info);

/**
 * @brief Free a specific DAG run op (MODELRUN/SCRIPTRUN).
 */
void RAI_DAGRunOpFree(RAI_DAGRunOp *dagOp);
