#ifndef SRC_MODEL_H_
#define SRC_MODEL_H_

#include "config.h"
#include "model_struct.h"
#include "tensor.h"
#include "redismodule.h"
#include "run_info.h"
#include "err.h"
#include "redisai.h"
#include "util/dict.h"
#include "run_info.h"

extern RedisModuleType *RedisAI_ModelType;

int RAI_ModelInit(RedisModuleCtx* ctx);
RAI_Model *RAI_ModelCreate(RAI_Backend backend, const char* devicestr, const char* tag, RAI_ModelOpts opts,
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
/* Return REDISMODULE_ERR if there was an error getting the Model.
 * Return REDISMODULE_OK if the model value stored at key was correctly
 * returned and available at *model variable. */
int RAI_GetModelFromKeyspace(RedisModuleCtx *ctx, RedisModuleString *keyName,
                              RedisModuleKey **key, RAI_Model **model,
                              int mode);

int RedisAI_Parse_ModelRun_RedisCommand(RedisModuleCtx *ctx,
                                        RedisModuleString **argv, int argc,
                                        RedisAI_RunInfo **rinfo,
                                        RAI_Model **mto, int useLocalContext,
                                        AI_dict **localContextDict, 
                                        int use_chaining_operator,
                                        const char *chaining_operator, RAI_Error *error);
#endif /* SRC_MODEL_H_ */
