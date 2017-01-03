#include "redismodule.h"
#include "tensorflow/c/c_api.h"
#include <string.h>

static RedisModuleType *RedisTF_TensorType;
static RedisModuleType *RedisTF_GraphType;

size_t RedisTF_DataSize(TF_DataType type) {
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

int RedisTF_SetValueFromDouble(TF_Tensor* tensor, long long i, double val) {
  switch (TF_TensorType(tensor)) {
    case TF_FLOAT:
      ((float*)TF_TensorData(tensor))[i] = val; break;
    case TF_DOUBLE:
      ((double*)TF_TensorData(tensor))[i] = val; break;
    default:
      return -1;
  }
  return 0;
}

int RedisTF_SetValueFromLongLong(TF_Tensor* tensor, long long i, long long val) {
  switch (TF_TensorType(tensor)) {
    case TF_BOOL:
      ((int8_t*)TF_TensorData(tensor))[i] = val; break;
    case TF_INT8:
      ((int8_t*)TF_TensorData(tensor))[i] = val; break;
    case TF_UINT8:
      ((uint8_t*)TF_TensorData(tensor))[i] = val; break;
    case TF_INT16:
      ((int16_t*)TF_TensorData(tensor))[i] = val; break;
    case TF_UINT16:
      ((uint16_t*)TF_TensorData(tensor))[i] = val; break;
    case TF_INT32:
      ((int32_t*)TF_TensorData(tensor))[i] = val; break;
    case TF_INT64:
      ((int64_t*)TF_TensorData(tensor))[i] = val; break;
    default:
      return -1;
  }
  return 0;
}

int RedisTF_GetValueAsDouble(TF_Tensor* tensor, long long i, double* val) {
  switch (TF_TensorType(tensor)) {
    case TF_FLOAT:
      *val = ((float*)TF_TensorData(tensor))[i]; break;
    case TF_DOUBLE:
      *val = ((double*)TF_TensorData(tensor))[i]; break;
    default:
      return -1;
  }
  return 0;
}

int RedisTF_GetValueAsLongLong(TF_Tensor* tensor, long long i, long long* val) {
  switch (TF_TensorType(tensor)) {
    case TF_BOOL:
      *val = ((int8_t*)TF_TensorData(tensor))[i]; break;
    case TF_INT8:
      *val = ((int8_t*)TF_TensorData(tensor))[i]; break;
    case TF_UINT8:
      *val = ((uint8_t*)TF_TensorData(tensor))[i]; break;
    case TF_INT16:
      *val = ((int16_t*)TF_TensorData(tensor))[i]; break;
    case TF_UINT16:
      *val = ((uint16_t*)TF_TensorData(tensor))[i]; break;
    case TF_INT32:
      *val = ((int32_t*)TF_TensorData(tensor))[i]; break;
    case TF_INT64:
      *val = ((int64_t*)TF_TensorData(tensor))[i]; break;
    default:
      return -1;
  }
  return 0;
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

struct RedisTF_TensorTypeObject {
  void *tensor;
};

struct RedisTF_TensorTypeObject *createTensorTypeObject(TF_Tensor* tensor) {
  struct RedisTF_TensorTypeObject *o = RedisModule_Alloc(sizeof(*o));
  o->tensor = tensor;
  return o;
}

struct RedisTF_GraphTypeObject {
  void *graph;
};

struct RedisTF_GraphTypeObject *createGraphTypeObject(TF_Graph* graph) {
  struct RedisTF_GraphTypeObject *o = RedisModule_Alloc(sizeof(*o));
  o->graph = graph;
  return o;
}

void RedisTF_TensorType_ReleaseObject(struct RedisTF_TensorTypeObject* o) {
  TF_DeleteTensor(o->tensor);
  RedisModule_Free(o);
}

void RedisTF_GraphType_ReleaseObject(struct RedisTF_GraphTypeObject* o) {
  TF_DeleteGraph(o->graph);
  RedisModule_Free(o);
}

enum RedisTF_DataFmt {
  REDISTF_DATA_BLOB = 0,
  REDISTF_DATA_VALUES
};

// ================================

// key type ndims dim1..dimN [BLOB data | VALUES val1..valN]
int RedisTF_Tensor_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 5) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],
      REDISMODULE_READ|REDISMODULE_WRITE);
  int type = RedisModule_KeyType(key);
  if (type != REDISMODULE_KEYTYPE_EMPTY &&
      RedisModule_ModuleTypeGetType(key) != RedisTF_TensorType) {
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

  if (argc < 4 + ndims) {
    return RedisModule_WrongArity(ctx);
  }

  long long len = 1;
  long long *dims = RedisModule_PoolAlloc(ctx, ndims * sizeof(long long));
  for (long long i=0; i<ndims; i++) {
    if ((RedisModule_StringToLongLong(argv[4+i], dims+i) != REDISMODULE_OK)
        || dims[i] < 0) {
      return RedisModule_ReplyWithError(ctx,"ERR invalid dims");
    }
    len *= dims[i];
  }

  if (argc != 4 + ndims && 
      argc != 4 + ndims + 1 + 1 &&
      argc != 4 + ndims + 1 + len) {
    return RedisModule_WrongArity(ctx);
  }

  int hasdata = argc > 4 + ndims;

  size_t fmtlen;
  const char* fmtstr;
  int datafmt;
  if (hasdata) {
    fmtstr = RedisModule_StringPtrLen(argv[5],&fmtlen);
    if (strcasecmp(fmtstr,"BLOB") == 0) datafmt = REDISTF_DATA_BLOB;
    else if (strcasecmp(fmtstr,"VALUES") == 0) datafmt = REDISTF_DATA_VALUES;
    else return RedisModule_ReplyWithError(ctx,"ERR unsupported data format");
  }

  size_t datasize = RedisTF_DataSize(datatype);
  if (datasize == 0) {
    return RedisModule_ReplyWithError(ctx,"ERR unsupported type");
  }
  len *= datasize;

  size_t datalen;
  const char* data;
  if (hasdata) {
    data = RedisModule_StringPtrLen(argv[6],&datalen);
    if (datafmt == REDISTF_DATA_BLOB && datalen != len) {
      return RedisModule_ReplyWithError(ctx,"ERR data length does not match tensor shape and type");
    }
  }

  if (hasdata && datafmt == REDISTF_DATA_VALUES && argc != len + 6) {
    return RedisModule_WrongArity(ctx);
  }

  TF_Tensor* tensor = TF_AllocateTensor((TF_DataType)datatype,dims,ndims,len);

  if (hasdata && datafmt == REDISTF_DATA_BLOB) {
    memcpy(TF_TensorData(tensor),data,datalen);
  }
  else if (hasdata && datafmt == REDISTF_DATA_VALUES) {
    long i;
    if (datatype == TF_FLOAT || datatype == TF_DOUBLE) {
      double val;
      for (i=0; i<len; i++) {
        if ((RedisModule_StringToDouble(argv[6+i], &val) != REDISMODULE_OK)) {
          TF_DeleteTensor(tensor);
          return RedisModule_ReplyWithError(ctx,"ERR invalid val");
        }
        int ret = RedisTF_SetValueFromDouble(tensor,i,val);
        if (ret == -1) {
          TF_DeleteTensor(tensor);
          return RedisModule_ReplyWithError(ctx,"ERR cannot specify values for this datatype");
        }
      }
    }
    else {
      long long val;
      for (i=0; i<len; i++) {
        if ((RedisModule_StringToLongLong(argv[6+i], &val) != REDISMODULE_OK)) {
          TF_DeleteTensor(tensor);
          return RedisModule_ReplyWithError(ctx,"ERR invalid val");
        }
        int ret = RedisTF_SetValueFromLongLong(tensor,i,val);
        if (ret == -1) {
          TF_DeleteTensor(tensor);
          return RedisModule_ReplyWithError(ctx,"ERR cannot specify values for this datatype");
        }
      }
    }
  }

  struct RedisTF_TensorTypeObject *tto = createTensorTypeObject(tensor);
  RedisModule_ModuleTypeSetValue(key,RedisTF_TensorType,tto);

  RedisModule_ReplyWithSimpleString(ctx, "OK");
  //RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}

int RedisTF_Type_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],
                                            REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (RedisModule_ModuleTypeGetType(key) != RedisTF_TensorType) {
    return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  struct RedisTF_TensorTypeObject *tto = RedisModule_ModuleTypeGetValue(key);

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

int RedisTF_NDims_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],
                                            REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (RedisModule_ModuleTypeGetType(key) != RedisTF_TensorType) {
    return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  struct RedisTF_TensorTypeObject *tto = RedisModule_ModuleTypeGetValue(key);

  long long ndims = TF_NumDims(tto->tensor);

  return RedisModule_ReplyWithLongLong(ctx,ndims);
}

int RedisTF_Dims_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],
                                            REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (RedisModule_ModuleTypeGetType(key) != RedisTF_TensorType) {
    return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  struct RedisTF_TensorTypeObject *tto = RedisModule_ModuleTypeGetValue(key);

  long long ndims = TF_NumDims(tto->tensor);

  RedisModule_ReplyWithArray(ctx,ndims);
  for (long long i=0; i<ndims; i++) {
    long long dim = TF_Dim(tto->tensor,i);
    RedisModule_ReplyWithLongLong(ctx,dim);
  }

  return REDISMODULE_OK;
}

int RedisTF_Size_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],
                                            REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (RedisModule_ModuleTypeGetType(key) != RedisTF_TensorType) {
    return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  struct RedisTF_TensorTypeObject *tto = RedisModule_ModuleTypeGetValue(key);

  long long size = TF_TensorByteSize(tto->tensor);

  return RedisModule_ReplyWithLongLong(ctx,size);
}

int RedisTF_Data_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],
                                            REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (RedisModule_ModuleTypeGetType(key) != RedisTF_TensorType) {
    return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  struct RedisTF_TensorTypeObject *tto = RedisModule_ModuleTypeGetValue(key);

  long long size = TF_TensorByteSize(tto->tensor);
  char *data = TF_TensorData(tto->tensor);

  return RedisModule_ReplyWithStringBuffer(ctx,data,size);
}

int RedisTF_Values_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 2) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],
                                            REDISMODULE_READ);
  int type = RedisModule_KeyType(key);
  if (RedisModule_ModuleTypeGetType(key) != RedisTF_TensorType) {
    return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  struct RedisTF_TensorTypeObject *tto = RedisModule_ModuleTypeGetValue(key);

  long long ndims = TF_NumDims(tto->tensor);
  long long len = 1;
  long long i;
  for (i=0; i<ndims; i++) {
    len *= TF_Dim(tto->tensor,i);
  }

  TF_DataType datatype = TF_TensorType(tto->tensor);

  RedisModule_ReplyWithArray(ctx,len);

  if (datatype == TF_FLOAT || datatype == TF_DOUBLE) {
    double val;
    for (i=0; i<len; i++) {
      int ret = RedisTF_GetValueAsDouble(tto->tensor,i,&val);
      if (ret == -1) {
        return RedisModule_ReplyWithError(ctx,"ERR cannot get values for this datatype");
      }
      RedisModule_ReplyWithDouble(ctx,val);
    }
  }
  else {
    long long val;
    for (i=0; i<len; i++) {
      int ret = RedisTF_GetValueAsLongLong(tto->tensor,i,&val);
      if (ret == -1) {
        return RedisModule_ReplyWithError(ctx,"ERR cannot get values for this datatype");
      }
      RedisModule_ReplyWithLongLong(ctx,val);
    }
  }

  return REDISMODULE_OK;
}


// ================================

// key graphbuf [prefix]
int RedisTF_Graph_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if ((argc != 3) && (argc != 4)) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],
      REDISMODULE_READ|REDISMODULE_WRITE);
  int type = RedisModule_KeyType(key);
  if (type != REDISMODULE_KEYTYPE_EMPTY &&
      RedisModule_ModuleTypeGetType(key) != RedisTF_GraphType) {
    return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  size_t graphlen;
  const char* graphdef = RedisModule_StringPtrLen(argv[2],&graphlen);

  TF_Graph* graph = TF_NewGraph();

  TF_ImportGraphDefOptions* options = TF_NewImportGraphDefOptions();

  if (argc == 4) {
    size_t prefixlen;
    const char* prefix = RedisModule_StringPtrLen(argv[3],&prefixlen);
    TF_ImportGraphDefOptionsSetPrefix(options, prefix);
  }
  else {
    TF_ImportGraphDefOptionsSetPrefix(options, "");
  }

  TF_Buffer *buffer = TF_NewBuffer();
  buffer->length = graphlen;
  buffer->data = graphdef;

  TF_Status *status = TF_NewStatus();

  TF_GraphImportGraphDef(graph,buffer,options,status);

  if (TF_GetCode(status) != TF_OK) {
    return RedisModule_ReplyWithError(ctx,TF_Message(status));
  }

  TF_DeleteImportGraphDefOptions(options);
  TF_DeleteBuffer(buffer);
  TF_DeleteStatus(status);

  struct RedisTF_GraphTypeObject *gto = createGraphTypeObject(graph);
  RedisModule_ModuleTypeSetValue(key,RedisTF_GraphType,gto);

  RedisModule_ReplyWithSimpleString(ctx, "OK");
  //RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}

// graph key, ninputs, (input key, input name)..., (output key, output name)...
int RedisTF_Run_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 3) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key = RedisModule_OpenKey(ctx,argv[1],REDISMODULE_READ);
  if (RedisModule_ModuleTypeGetType(key) != RedisTF_GraphType) {
    return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  // Get the graph

  struct RedisTF_GraphTypeObject *gto = RedisModule_ModuleTypeGetValue(key);

  TF_Graph* graph = gto->graph;

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

  TF_Status *status = TF_NewStatus();

  // Build input/output ports (TODO: targets)

  TF_Port *inports = RedisModule_PoolAlloc(ctx, ninputs*sizeof(TF_Port));
  TF_Port *outports = RedisModule_PoolAlloc(ctx, noutputs*sizeof(TF_Port));
  TF_Tensor **intensors = RedisModule_PoolAlloc(ctx, ninputs*sizeof(TF_Tensor*));
  TF_Tensor **outtensors = RedisModule_PoolAlloc(ctx, noutputs*sizeof(TF_Tensor*));

  for (int i=pairoffset; i<argc; i+=2) {
    int isinput = i < pairoffset + 2 * ninputs;
    int flags = isinput ? REDISMODULE_READ : REDISMODULE_READ|REDISMODULE_WRITE;

    RedisModuleKey *argkey = RedisModule_OpenKey(ctx,argv[i],flags);
    int argtype = RedisModule_KeyType(argkey);

    size_t namelen;
    RedisModuleString* argname = argv[i+1];

    if (isinput) {
      if (RedisModule_ModuleTypeGetType(argkey) != RedisTF_TensorType) {
        return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
      }
      struct RedisTF_TensorTypeObject *tto = RedisModule_ModuleTypeGetValue(argkey);
      intensors[(i-pairoffset)/2] = clone(ctx,tto->tensor);
      const char* opname = RedisModule_StringPtrLen(argname,&namelen);
      RedisModule_Log(ctx,"warning","%s",opname);
      TF_Port port;
      port.oper = TF_GraphOperationByName(graph,opname);
      port.index = 0;
      inports[(i-pairoffset)/2] = port;
    } else {
      const char* opname = RedisModule_StringPtrLen(argname,&namelen);
      TF_Port port;
      port.oper = TF_GraphOperationByName(graph,opname);
      port.index = 0;
      outports[(i-pairoffset)/2-ninputs] = port;
    }
  }

  // TODO: create target ops, e.g.:
  //TF_Operation *init = TF_GraphOperationByName(graph,"init_op");
  // We need to change the command to allow for specific targets to be specified by name

  // Create session and run the graph

  TF_SessionOptions *options = TF_NewSessionOptions();
  TF_Session *session = TF_NewSession(graph,options,status);

  if (TF_GetCode(status) != TF_OK) {
    return RedisModule_ReplyWithError(ctx,TF_Message(status));
  }

  TF_SessionRun(session,NULL,
                inports, intensors, ninputs,
                outports, outtensors, noutputs,
                NULL,0,
                NULL,
                status);
  if (TF_GetCode(status) != TF_OK) {
    return RedisModule_ReplyWithError(ctx,TF_Message(status));
  }

  for (int i=0; i<noutputs; i++) {
    RedisModuleKey *outkey = RedisModule_OpenKey(ctx,argv[pairoffset+2*ninputs+2*i],
                                                 REDISMODULE_READ|REDISMODULE_WRITE);
    int type = RedisModule_KeyType(outkey);
    if (type != REDISMODULE_KEYTYPE_EMPTY &&
        RedisModule_ModuleTypeGetType(outkey) != RedisTF_TensorType) {
      return RedisModule_ReplyWithError(ctx,REDISMODULE_ERRORMSG_WRONGTYPE);
    }
    struct RedisTF_TensorTypeObject *tto = createTensorTypeObject(outtensors[i]);
    RedisModule_ModuleTypeSetValue(outkey,RedisTF_TensorType,tto);
  }

  TF_CloseSession(session,status);

  TF_DeleteSession(session,status);
  TF_DeleteSessionOptions(options);
  TF_DeleteStatus(status);

  RedisModule_ReplyWithSimpleString(ctx, "OK");
  //RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}


// ================================

void *RedisTF_TensorType_RdbLoad(RedisModuleIO *rdb, int encver) {
  //if (encver != 0) {
  //  /* RedisModule_Log("warning","Can't load data with version %d", encver);*/
  //  return NULL;
  //}
  // TODO
  return NULL;
}

void RedisTF_TensorType_RdbSave(RedisModuleIO *rdb, void *value) {
  // TODO
}

void RedisTF_TensorType_AofRewrite(RedisModuleIO *aof, RedisModuleString *key, void *value) {
  // TODO
}

void RedisTF_TensorType_Digest(RedisModuleDigest *digest, void *value) {
  // TODO
}

void RedisTF_TensorType_Free(void *value) {
  RedisTF_TensorType_ReleaseObject(value);
}

// ================================

void *RedisTF_GraphType_RdbLoad(RedisModuleIO *rdb, int encver) {
  //if (encver != 0) {
  //  /* RedisModule_Log("warning","Can't load data with version %d", encver);*/
  //  return NULL;
  //}
  // TODO
  return NULL;
}

void RedisTF_GraphType_RdbSave(RedisModuleIO *rdb, void *value) {
  // TODO
}

void RedisTF_GraphType_AofRewrite(RedisModuleIO *aof, RedisModuleString *key, void *value) {
  // TODO
}

void RedisTF_GraphType_Digest(RedisModuleDigest *digest, void *value) {
  // TODO
}

void RedisTF_GraphType_Free(void *value) {
  RedisTF_GraphType_ReleaseObject(value);
}


int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {

  if (RedisModule_Init(ctx,"tf",1,REDISMODULE_APIVER_1)
      == REDISMODULE_ERR) return REDISMODULE_ERR;

  RedisTF_TensorType = RedisModule_CreateDataType(ctx,"TF_TENSOR",0,
                                                 RedisTF_TensorType_RdbLoad,
                                                 RedisTF_TensorType_RdbSave,
                                                 RedisTF_TensorType_AofRewrite,
                                                 RedisTF_TensorType_Digest,
                                                 RedisTF_TensorType_Free);
  if (RedisTF_TensorType == NULL) return REDISMODULE_ERR;

  RedisTF_GraphType = RedisModule_CreateDataType(ctx,"TF_TENSOR",0,
                                                RedisTF_GraphType_RdbLoad,
                                                RedisTF_GraphType_RdbSave,
                                                RedisTF_GraphType_AofRewrite,
                                                RedisTF_GraphType_Digest,
                                                RedisTF_GraphType_Free);
  if (RedisTF_GraphType == NULL) return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx,"tf.tensor",RedisTF_Tensor_RedisCommand,"write",1,1,1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx,"tf.type",RedisTF_Type_RedisCommand,"readonly",1,1,1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx,"tf.ndims",RedisTF_NDims_RedisCommand,"readonly",1,1,1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx,"tf.dims",RedisTF_Dims_RedisCommand,"readonly",1,1,1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx,"tf.size",RedisTF_Size_RedisCommand,"readonly",1,1,1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx,"tf.data",RedisTF_Data_RedisCommand,"readonly",1,1,1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx,"tf.values",RedisTF_Values_RedisCommand,"readonly",1,1,1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx,"tf.graph",RedisTF_Graph_RedisCommand,"write",1,1,1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx,"tf.run",RedisTF_Run_RedisCommand,"write",1,-1,2)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  return REDISMODULE_OK;
}

