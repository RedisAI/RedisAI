#pragma once

#include "redisai.h"

RAI_DAGRunCtx *RAI_DAGRunCtxCreate(void);

RAI_DAGRunOp *RAI_DAGCreateModelRunOp(RAI_Model *model);

RAI_DAGRunOp *RAI_DAGCreateScriptRunOp(RAI_Script *script, const char *func_name);

int RAI_DAGRunOpAddInput(RAI_DAGRunOp *DAGOp, const char *input);

int RAI_DAGRunOpAddOutput(RAI_DAGRunOp *DAGOp, const char *output);

int RAI_DAGAddRunOp(RAI_DAGRunCtx *run_info, RAI_DAGRunOp *DAGop, RAI_Error *err);

int RAI_DAGLoadTensor(RAI_DAGRunCtx *run_info, const char *t_name, RAI_Error *err);

int RAI_DAGLoadTensorRS(RAI_DAGRunCtx *run_info, RedisModuleString *t_name, RAI_Error *err);

int RAI_DAGAddTensorSet(RAI_DAGRunCtx *run_info, const char *t_name, RAI_Tensor *tensor);

int RAI_DAGAddTensorGet(RAI_DAGRunCtx *run_info, const char *t_name, RAI_Error *err);

size_t RAI_DAGNumOps(RAI_DAGRunCtx *run_info);

void RAI_DAGFree(RAI_DAGRunCtx *run_info);

void RAI_DAGRunOpFree(RAI_DAGRunOp *dagOp);