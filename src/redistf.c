#include "redismodule.h"
#include "tensorflow/c/c_api.h"
#include <string.h>

static RedisModuleType *TFTensorType;
#if 0
static RedisModuleType *TFGraphType;
#endif

size_t TFDataSize(TF_DataType type) {
  switch (type) {
  case TF_BOOL:
    return sizeof(int8_t);
  case TF_INT8:
    return sizeof(int8_t);
  case TF_UINT8:
    return sizeof(uint8_t);
  case TF_INT16:
    return sizeof(int16_t);
  case TF_UINT16:
    return sizeof(uint16_t);
  case TF_INT32:
    return sizeof(int32_t);
  case TF_INT64:
    return sizeof(int64_t);
  case TF_FLOAT:
    return sizeof(float);
  case TF_DOUBLE:
    return sizeof(double);
  case TF_COMPLEX64:
    return 2*sizeof(float);
  case TF_COMPLEX128:
    return 2 * sizeof(double);
  default:
    return 0;
  }
}

TF_Tensor* clone(RedisModuleCtx *ctx, const TF_Tensor *tensor) {
  int ndims = TF_NumDims(tensor);
  long long *dims = RedisModule_PoolAlloc(ctx, ndims * sizeof(long long));
  for (int j=0; j<ndims; j++) {
    dims[j] = TF_Dim(tensor,j);
  }
  size_t len = TF_TensorByteSize(tensor);
  void *data = TF_TensorData(tensor);
  TF_Tensor *out = TF_AllocateTensor(TF_TensorType(tensor),
                                     dims,ndims,len);
  memcpy(TF_TensorData(out),data,len);
  return out;
}

struct TFTensorTypeObject {
  void *tensor;
};

struct TFTensorTypeObject *createTensorTypeObject(TF_Tensor* tensor) {
  struct TFTensorTypeObject *o = RedisModule_Alloc(sizeof(*o));
  o->tensor = tensor;
  return o;
}

#if 0
struct TFGraphTypeObject {
  void *graph;
};

struct TFGraphTypeObject *createGraphTypeObject(TF_Graph* graph) {
  struct TFGraphTypeObject *o = RedisModule_Alloc(sizeof(*o));
  o->graph = graph;
  return o;
}
#endif

void TFTensorTypeReleaseObject(struct TFTensorTypeObject* o) {
  TF_DeleteTensor(o->tensor);
  RedisModule_Free(o);
}

#if 0
void TFGraphTypeReleaseObject(struct TFGraphTypeObject* o) {

  TF_DeleteGraph(o->graph);
  RedisModule_Free(o);
}
#endif


// ================================

// key, type, ndims, dims, [data]
int TF_Tensor_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 4) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],
      REDISMODULE_READ|REDISMODULE_WRITE);
  int type = RedisModule_KeyType(key);
  if (type != REDISMODULE_KEYTYPE_EMPTY &&
      RedisModule_ModuleTypeGetType(key) != TFTensorType) {
    return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  size_t typelen;
  TF_DataType datatype;
  char typerr = 0;
  const char* typestr = RedisModule_StringPtrLen(argv[2],&typelen);
  if (strcasecmp(typestr,"FLOAT") == 0) datatype = TF_FLOAT;
  else if (strcasecmp(typestr,"DOUBLE") == 0) datatype = TF_DOUBLE;
  else if (strncasecmp(typestr,"INT",3) == 0) {
    const char *bitstr = typestr + 3;
    if (strcmp(bitstr,"8") == 0) datatype = TF_INT8;
    else if (strcmp(bitstr,"16") == 0) datatype = TF_INT16;
    else if (strcmp(bitstr,"32") == 0) datatype = TF_INT32;
    else if (strcmp(bitstr,"64") == 0) datatype = TF_INT64;
    else typerr = 1;
  }
  else if (strncasecmp(typestr,"UINT",4) == 0) {
    const char *bitstr = typestr + 4;
    if (strcmp(bitstr,"8") == 0) datatype = TF_UINT8;
    else if (strcmp(bitstr,"16") == 0) datatype = TF_UINT16;
    else typerr = 1;
  }
  else typerr = 1;

  if (typerr) {
    return RedisModule_ReplyWithError(ctx,"ERR invalid type");
  }

  long long ndims = 0;
  if ((RedisModule_StringToLongLong(argv[3], &ndims) != REDISMODULE_OK)
      || ndims < 0) {
    return RedisModule_ReplyWithError(ctx,"ERR invalid ndims");
  }

  if (argc != 4 + ndims && argc != 4 + ndims + 1) {
    return RedisModule_WrongArity(ctx);
  }

  int hasdata = argc == 4 + ndims + 1;

  long long len = 1;
  long long *dims = RedisModule_PoolAlloc(ctx, ndims * sizeof(long long));
  for (long long i=0; i<ndims; i++) {
    if ((RedisModule_StringToLongLong(argv[4+i], dims+i) != REDISMODULE_OK)
        || dims[i] < 0) {
      return RedisModule_ReplyWithError(ctx,"ERR invalid dims");
    }
    len *= dims[i];
  }

  size_t datasize = TFDataSize(datatype);
  if (datasize == 0) {
    return RedisModule_ReplyWithError(ctx,"ERR unsupported type");
  }
  len *= datasize;

  size_t datalen;
  const char* data;
  if (hasdata) {
    data = RedisModule_StringPtrLen(argv[5],&datalen);
    if (datalen != len) {
      return RedisModule_ReplyWithError(ctx,"ERR data length does not match tensor shape and type");
    }
  }

  TF_Tensor* tensor = TF_AllocateTensor((TF_DataType)datatype,dims,ndims,len);

  if (hasdata) {
    memcpy(TF_TensorData(tensor),data,datalen);
  }

  struct TFTensorTypeObject *tto = createTensorTypeObject(tensor);
  RedisModule_ModuleTypeSetValue(key,TFTensorType,tto);

  RedisModule_ReplyWithSimpleString(ctx, "OK");
  //RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}

int TF_Type_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],
                                            REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (RedisModule_ModuleTypeGetType(key) != TFTensorType) {
    return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  struct TFTensorTypeObject *tto = RedisModule_ModuleTypeGetValue(key);

  //RedisModule_Log(ctx,"warning","%s",THByteTensor_desc(tto->tensor).str);

  switch (TF_TensorType(tto->tensor)) {
  case TF_FLOAT:
    return RedisModule_ReplyWithSimpleString(ctx,"FLOAT");
  case TF_DOUBLE:
    return RedisModule_ReplyWithSimpleString(ctx,"DOUBLE");
  case TF_INT8:
    return RedisModule_ReplyWithSimpleString(ctx,"INT8");
  case TF_INT16:
    return RedisModule_ReplyWithSimpleString(ctx,"INT16");
  case TF_INT32:
    return RedisModule_ReplyWithSimpleString(ctx,"INT32");
  case TF_INT64:
    return RedisModule_ReplyWithSimpleString(ctx,"INT64");
  case TF_UINT8:
    return RedisModule_ReplyWithSimpleString(ctx,"UINT8");
  case TF_UINT16:
    return RedisModule_ReplyWithSimpleString(ctx,"UINT16");
  default:
    return RedisModule_ReplyWithError(ctx,"ERR unsupported type");
  }
}

int TF_NDims_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],
                                            REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (RedisModule_ModuleTypeGetType(key) != TFTensorType) {
    return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  struct TFTensorTypeObject *tto = RedisModule_ModuleTypeGetValue(key);

  long long ndims = TF_NumDims(tto->tensor);

  return RedisModule_ReplyWithLongLong(ctx,ndims);
}

int TF_Dims_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],
                                            REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (RedisModule_ModuleTypeGetType(key) != TFTensorType) {
    return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  struct TFTensorTypeObject *tto = RedisModule_ModuleTypeGetValue(key);

  long long ndims = TF_NumDims(tto->tensor);

  RedisModule_ReplyWithArray(ctx,ndims);
  for (long long i=0; i<ndims; i++) {
    long long dim = TF_Dim(tto->tensor,i);
    RedisModule_ReplyWithLongLong(ctx,dim);
  }

  return REDISMODULE_OK;
}

int TF_Size_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],
                                            REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (RedisModule_ModuleTypeGetType(key) != TFTensorType) {
    return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  struct TFTensorTypeObject *tto = RedisModule_ModuleTypeGetValue(key);

  long long size = TF_TensorByteSize(tto->tensor);

  return RedisModule_ReplyWithLongLong(ctx,size);
}

int TF_Data_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],
                                            REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (RedisModule_ModuleTypeGetType(key) != TFTensorType) {
    return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  struct TFTensorTypeObject *tto = RedisModule_ModuleTypeGetValue(key);

  long long size = TF_TensorByteSize(tto->tensor);
  char *data = TF_TensorData(tto->tensor);

  return RedisModule_ReplyWithStringBuffer(ctx,data,size);
}

// ================================

#if 0
int TF_Graph_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],
      REDISMODULE_READ|REDISMODULE_WRITE);
  int type = RedisModule_KeyType(key);
  if (type != REDISMODULE_KEYTYPE_EMPTY &&
      RedisModule_ModuleTypeGetType(key) != TFGraphType) {
    return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  TF_Graph* graph = TF_NewGraph();

  struct TFGraphTypeObject *gto = createGraphTypeObject(graph);
  RedisModule_ModuleTypeSetValue(key,TFGraphType,gto);

  RedisModule_ReplyWithSimpleString(ctx, "OK");
  //RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}
#endif

// graph, ninputs, (input key, input name)..., (output key, output name)...
int TF_Run_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  //RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],
  //                                          REDISMODULE_READ|REDISMODULE_WRITE);
  //int type = RedisModule_KeyType(key);
  //if (type != REDISMODULE_KEYTYPE_EMPTY &&
  //    RedisModule_ModuleTypeGetType(key) != TFGraphType)
  //  {
  //    return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
  //  }

  //struct TFGraphTypeObject *gto = RedisModule_ModuleTypeGetValue(key);

  //TF_Graph* graph = gto->graph;

  RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (RedisModule_KeyType(key) != REDISMODULE_KEYTYPE_STRING) {
    return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  long long ninputs;
  if ((RedisModule_StringToLongLong(argv[2], &ninputs) != REDISMODULE_OK)
      || ninputs < 0) {
    return RedisModule_ReplyWithError(ctx,"ERR invalid ninputs");
  }

  int pairoffset = 3;
  int npairs = (argc - pairoffset) / 2;
  int noutputs = npairs - ninputs;

  if (npairs < ninputs) {
    return RedisModule_ReplyWithError(ctx,"ERR key/name pairs less than ninputs");
  }

  if ((argc - pairoffset) % 2 != 0) {
    return RedisModule_ReplyWithError(ctx,"ERR odd key/name pairs");
  }

  TF_Tensor **intensors = RedisModule_PoolAlloc(ctx, ninputs*sizeof(TF_Tensor*));
  TF_Tensor **outtensors = RedisModule_PoolAlloc(ctx, noutputs*sizeof(TF_Tensor*));

  const char **innames = RedisModule_PoolAlloc(ctx, ninputs*sizeof(char*));
  const char **outnames = RedisModule_PoolAlloc(ctx, noutputs*sizeof(char*));

  for (int i=pairoffset; i<argc; i+=2) {
    int isinput = i < pairoffset + 2 * ninputs;
    int flags = isinput ? REDISMODULE_READ : REDISMODULE_READ|REDISMODULE_WRITE;

    RedisModuleKey *argkey = RedisModule_OpenKey(ctx,argv[i],flags);
    int argtype = RedisModule_KeyType(argkey);

    size_t namelen;
    RedisModuleString* argname = argv[i+1];

    if (isinput) {
      if (RedisModule_ModuleTypeGetType(argkey) != TFTensorType) {
        return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
      }
      struct TFTensorTypeObject *tto = RedisModule_ModuleTypeGetValue(argkey);
      intensors[(i-pairoffset)/2] = clone(ctx,tto->tensor);
      innames[(i-pairoffset)/2] = RedisModule_StringPtrLen(argname,&namelen);
    } else {
      outnames[(i-pairoffset)/2-ninputs] = RedisModule_StringPtrLen(argname,&namelen);
    }
  }

  TF_SessionOptions *options = TF_NewSessionOptions();
  TF_Status *status = TF_NewStatus();

  TF_Session *session = TF_NewSession(options,status);

  size_t graphlen;
  const char* graphdef = RedisModule_StringDMA(key,&graphlen,REDISMODULE_READ);

  TF_ExtendGraph(session,graphdef,graphlen,status);
  if (TF_GetCode(status) != TF_OK) {
    return RedisModule_ReplyWithError(ctx,TF_Message(status));
  }

  TF_Run(session,NULL,
         innames,intensors,ninputs,
         outnames,outtensors,noutputs,
         NULL,0,NULL,status);
  if (TF_GetCode(status) != TF_OK) {
    return RedisModule_ReplyWithError(ctx,TF_Message(status));
  }

  for (int i=0; i<noutputs; i++) {
    RedisModuleKey *outkey = RedisModule_OpenKey(ctx,argv[pairoffset+2*ninputs+2*i],
                                                 REDISMODULE_READ|REDISMODULE_WRITE);
    int type = RedisModule_KeyType(outkey);
    if (type != REDISMODULE_KEYTYPE_EMPTY &&
        RedisModule_ModuleTypeGetType(outkey) != TFTensorType) {
      return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
    }
    struct TFTensorTypeObject *tto = createTensorTypeObject(outtensors[i]);
    RedisModule_ModuleTypeSetValue(outkey,TFTensorType,tto);
  }

  TF_CloseSession(session,status);
  TF_DeleteStatus(status);
  TF_DeleteSessionOptions(options);
  TF_DeleteSession(session);

  RedisModule_ReplyWithSimpleString(ctx, "OK");
  //RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}


// ================================

void *TFTensorTypeRdbLoad(RedisModuleIO *rdb, int encver) {
  //if (encver != 0) {
  //  /* RedisModule_Log("warning","Can't load data with version %d", encver);*/
  //  return NULL;
  //}
  // TODO
  return NULL;
}

void TFTensorTypeRdbSave(RedisModuleIO *rdb, void *value) {
  // TODO
}

void TFTensorTypeAofRewrite(RedisModuleIO *aof, RedisModuleString *key, void *value) {
  // TODO
}

void TFTensorTypeDigest(RedisModuleDigest *digest, void *value) {
  // TODO
}

void TFTensorTypeFree(void *value) {
  TFTensorTypeReleaseObject(value);
}

// ================================

#if 0
void *TFGraphTypeRdbLoad(RedisModuleIO *rdb, int encver) {
  //if (encver != 0) {
  //  /* RedisModule_Log("warning","Can't load data with version %d", encver);*/
  //  return NULL;
  //}
  // TODO
  return NULL;
}

void TFGraphTypeRdbSave(RedisModuleIO *rdb, void *value) {
  // TODO
}

void TFGraphTypeAofRewrite(RedisModuleIO *aof, RedisModuleString *key, void *value) {
  // TODO
}

void TFGraphTypeDigest(RedisModuleDigest *digest, void *value) {
  // TODO
}

void TFGraphTypeFree(void *value) {
  TFGraphTypeReleaseObject(value);
}
#endif


int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {

  if (RedisModule_Init(ctx,"tf",1,REDISMODULE_APIVER_1)
      == REDISMODULE_ERR) return REDISMODULE_ERR;

  TFTensorType = RedisModule_CreateDataType(ctx,"TF_TENSOR",0,TFTensorTypeRdbLoad,
                                            TFTensorTypeRdbSave,TFTensorTypeAofRewrite,
                                            TFTensorTypeDigest,TFTensorTypeFree);
  if (TFTensorType == NULL) return REDISMODULE_ERR;

#if 0
  TFGraphType = RedisModule_CreateDataType(ctx,"TF_TENSOR",0,TFGraphTypeRdbLoad,
                                            TFGraphTypeRdbSave,TFGraphTypeAofRewrite,
                                            TFGraphTypeDigest,TFGraphTypeFree);
  if (TFGraphType == NULL) return REDISMODULE_ERR;
#endif

  if (RedisModule_CreateCommand(ctx,"tf.tensor",TF_Tensor_RedisCommand,"write",1,1,1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx,"tf.type",TF_Type_RedisCommand,"readonly",1,1,1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx,"tf.ndims",TF_NDims_RedisCommand,"readonly",1,1,1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx,"tf.dims",TF_Dims_RedisCommand,"readonly",1,1,1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx,"tf.size",TF_Size_RedisCommand,"readonly",1,1,1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx,"tf.data",TF_Data_RedisCommand,"readonly",1,1,1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

#if 0
  if (RedisModule_CreateCommand(ctx,"tf.graph",TF_Graph_RedisCommand,"write",1,1,1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;
#endif

  if (RedisModule_CreateCommand(ctx,"tf.run",TF_Run_RedisCommand,"write",1,-1,2)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  return REDISMODULE_OK;
}

