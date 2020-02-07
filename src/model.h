#ifndef SRC_MODEL_H_
#define SRC_MODEL_H_

#include "config.h"
#include "model_struct.h"
#include "tensor.h"
#include "redismodule.h"
#include "err.h"

extern RedisModuleType *RedisAI_ModelType;

int RAI_ModelInit(RedisModuleCtx* ctx);
RAI_Model *RAI_ModelCreate(RAI_Backend backend, const char* devicestr, RAI_ModelOpts opts,
                           size_t ninputs, const char **inputs,
                           size_t noutputs, const char **outputs,
                           const char *modeldef, size_t modellen, RAI_Error* err);
void RAI_ModelFree(RAI_Model* model, RAI_Error* err);

RAI_ModelRunCtx* RAI_ModelRunCtxCreate(RAI_Model* model);

int RAI_ModelRunCtxAddBatch(RAI_ModelRunCtx* mctx);
size_t RAI_ModelRunCtxNumBatches(RAI_ModelRunCtx* mctx);
void RAI_ModelRunCtxCopyBatch(RAI_ModelRunCtx* dest, size_t id_dest, RAI_ModelRunCtx* src, size_t id_src);
int RAI_ModelRunCtxAddInput(RAI_ModelRunCtx* mctx, size_t id, const char* inputName, RAI_Tensor* inputTensor);
int RAI_ModelRunCtxAddOutput(RAI_ModelRunCtx* mctx, size_t id, const char* outputName);
size_t RAI_ModelRunCtxNumInputs(RAI_ModelRunCtx* mctx);
size_t RAI_ModelRunCtxNumOutputs(RAI_ModelRunCtx* mctx);
RAI_Tensor* RAI_ModelRunCtxInputTensor(RAI_ModelRunCtx* mctx, size_t id, size_t index);
RAI_Tensor* RAI_ModelRunCtxOutputTensor(RAI_ModelRunCtx* mctx, size_t id, size_t index);
void RAI_ModelRunCtxFree(RAI_ModelRunCtx* mctx);

int RAI_ModelRun(RAI_ModelRunCtx* mctx, RAI_Error* err);
RAI_Model* RAI_ModelGetShallowCopy(RAI_Model* model);

int RAI_ModelSerialize(RAI_Model *model, char **buffer, size_t *len, RAI_Error *err);

#endif /* SRC_MODEL_H_ */
