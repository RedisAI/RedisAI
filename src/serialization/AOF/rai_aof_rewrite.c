#include "rai_aof_rewrite.h"

void RAI_AofRewriteTensor(RedisModuleIO *aof, RedisModuleString *key, void *value) {
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