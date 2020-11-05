#include "rai_aof_rewrite.h"

void RAI_AOFRewriteTensor(RedisModuleIO *aof, RedisModuleString *key, void *value) {
  RAI_Tensor *tensor = (RAI_Tensor*)value;

  char *dtypestr = NULL;
  Tensor_DataTypeStr(RAI_TensorDataType(tensor), &dtypestr);

  char *data = RAI_TensorData(tensor);
  long long size = RAI_TensorByteSize(tensor);

  long long ndims = RAI_TensorNumDims(tensor);

  RedisModuleString* dims[ndims];

  for (long long i=0; i<ndims; i++) {
    dims[i] = RedisModule_CreateStringFromLongLong(RedisModule_GetContextFromIO(aof), RAI_TensorDim(tensor, i));
  }

  RedisModule_EmitAOF(aof, "AI.TENSORSET", "scvcb", key, dtypestr, dims, ndims, "BLOB", data, size);
 
  RedisModule_Free(dtypestr);
}

void RAI_AOFRewriteModel(RedisModuleIO *aof, RedisModuleString *key, void *value) {
  RAI_Model *model = (RAI_Model*)value;

  char *buffer = NULL;
  size_t len = 0;
  RAI_Error err = {0};

  int ret = RAI_ModelSerialize(model, &buffer, &len, &err);

  if (err.code != RAI_OK) {
    
    printf("ERR: %s\n", err.detail);
    RAI_ClearError(&err);
    if (buffer) {
      RedisModule_Free(buffer);
    }
    return;
  }

  // AI.MODELSET model_key backend device [INPUTS name1 name2 ... OUTPUTS name1 name2 ...] model_blob

  RedisModuleString **inputs_ = array_new(RedisModuleString*, model->ninputs);
  RedisModuleString **outputs_ = array_new(RedisModuleString*, model->noutputs);

  RedisModuleCtx *ctx = RedisModule_GetContextFromIO(aof);

  for (size_t i=0; i<model->ninputs; i++) {
    inputs_ = array_append(inputs_, RedisModule_CreateString(ctx, model->inputs[i], strlen(model->inputs[i])));
  }

  for (size_t i=0; i<model->noutputs; i++) {
    outputs_ = array_append(outputs_, RedisModule_CreateString(ctx, model->outputs[i], strlen(model->outputs[i])));
  }

  long long chunk_size = getModelChunkSize();
  const size_t n_chunks = len / chunk_size + 1;
  RedisModuleString **buffers_ = array_new(RedisModuleString*, n_chunks);

  for (size_t i=0; i<n_chunks; i++) {
    size_t chunk_len = i < n_chunks - 1 ? chunk_size : len % chunk_size;
    buffers_ = array_append(buffers_, RedisModule_CreateString(ctx, buffer + i * chunk_size, chunk_len));
  }

  if (buffer) {
    RedisModule_Free(buffer);
  }

  const char* backendstr = RAI_BackendName(model->backend);

  RedisModule_EmitAOF(aof, "AI.MODELSET", "slccclclcvcvcv",
                      key,
                      backendstr, model->devicestr, model->tag,
                      "BATCHSIZE", model->opts.batchsize,
                      "MINBATCHSIZE", model->opts.minbatchsize,
                      "INPUTS", inputs_, model->ninputs,
                      "OUTPUTS", outputs_, model->noutputs,
                      "BLOB", buffers_, n_chunks);

  for (size_t i=0; i<model->ninputs; i++) {
    RedisModule_FreeString(ctx, inputs_[i]);
  }
  array_free(inputs_);

  for (size_t i=0; i<model->noutputs; i++) {
    RedisModule_FreeString(ctx, outputs_[i]);
  }
  array_free(outputs_);

  for (size_t i=0; i<n_chunks; i++) {
    RedisModule_FreeString(ctx, buffers_[i]);
  }
  array_free(buffers_);
}

void RAI_AOFRewriteScript(RedisModuleIO *aof, RedisModuleString *key, void *value){  
  RAI_Script* script = (RAI_Script*)value;
  RedisModule_EmitAOF(aof, "AI.SCRIPTSET", "scccc", key, script->devicestr,
                      script->tag, "SOURCE", script->scriptdef);
}