#include "redismodule.h"
#include "tensor.h"

#include "model.h"
#include "dag.h"
#include "model_script_run_session.h"
#include "background_workers.h"
#include "script.h"
#include "backends.h"
#include "stats.h"
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdbool.h>
#include "backends/util.h"

#include "rmutil/alloc.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"
#include "util/queue.h"
#include "rmutil/args.h"
#include "run_info.h"
#include "version.h"

#define REDISAI_H_INCLUDE
#include "redisai.h"
#undef REDISAI_H_INCLUDE

#ifndef REDISAI_GIT_SHA
#define REDISAI_GIT_SHA "unknown"
#endif

int redisMajorVersion;
int redisMinorVersion;
int redisPatchVersion;

int rlecMajorVersion;
int rlecMinorVersion;
int rlecPatchVersion;
int rlecBuild;

void getRedisVersion() {
    RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(NULL);
    RedisModuleCallReply *reply = RedisModule_Call(ctx, "info", "c", "server");
    assert(RedisModule_CallReplyType(reply) == REDISMODULE_REPLY_STRING);
    size_t len;
    const char *replyStr = RedisModule_CallReplyStringPtr(reply, &len);

    int n = sscanf(replyStr, "# Server\nredis_version:%d.%d.%d", &redisMajorVersion,
                 &redisMinorVersion, &redisPatchVersion);

    assert(n == 3);

    rlecMajorVersion = -1;
    rlecMinorVersion = -1;
    rlecPatchVersion = -1;
    rlecBuild = -1;
    char *enterpriseStr = strstr(replyStr, "rlec_version:");
    if (enterpriseStr) {
        n = sscanf(enterpriseStr, "rlec_version:%d.%d.%d-%d", &rlecMajorVersion, &rlecMinorVersion,
                   &rlecPatchVersion, &rlecBuild);
        if (n != 4) {
            RedisModule_Log(NULL, "warning", "Could not extract enterprise version");
        }
    }

    RedisModule_FreeCallReply(reply);
    RedisModule_FreeThreadSafeContext(ctx);
}

static inline int IsEnterprise() {
  return rlecMajorVersion != -1;
}

/* ----------------------- RedisAI Module Commands ------------------------- */

/**
 * AI.TENSORSET key type dim1..dimN [BLOB data | VALUES val1..valN]
 */
int RedisAI_TensorSet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc < 4) return RedisModule_WrongArity(ctx);

  RedisModuleKey *key;
  const int status = RAI_OpenKey_Tensor(ctx, argv[1], &key, REDISMODULE_READ|REDISMODULE_WRITE);
  if(status==REDISMODULE_ERR){
      return REDISMODULE_ERR;
  }

  RAI_Tensor *t=NULL;
  RAI_Error err;
  const int parse_result = RAI_parseTensorSetArgs(ctx,argv,argc,&t,1,&err);

  // if the number of parsed args is negative something went wrong
  if(parse_result<0){
    RedisModule_CloseKey(key);
    return REDISMODULE_ERR;
  }

  if( RedisModule_ModuleTypeSetValue(key, RedisAI_TensorType, t) != REDISMODULE_OK ){
    RAI_TensorFree(t);
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR could not save tensor");
  }
  RedisModule_CloseKey(key);
  RedisModule_ReplyWithSimpleString(ctx, "OK");
  RedisModule_ReplicateVerbatim(ctx);
  return REDISMODULE_OK;
}

/**
* AI.TENSORGET tensor_key [META] [BLOB | VALUES]
*/
int RedisAI_TensorGet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc < 2 || argc > 4) return RedisModule_WrongArity(ctx);

  RAI_Tensor *t;
  RedisModuleKey *key;
  const int status = RAI_GetTensorFromKeyspace(ctx, argv[1], &key, &t, REDISMODULE_READ);
  if(status==REDISMODULE_ERR){
  return REDISMODULE_ERR;
  }

  const int parse_result = RAI_parseTensorGetArgs(ctx, argv, argc, t);

  RedisModule_CloseKey(key);
  // if the number of parsed args is negative something went wrong
  if (parse_result < 0) {
    return REDISMODULE_ERR;
  }
  return REDISMODULE_OK;
}

/**
* AI.MODELSET model_key backend device [TAG tag] [BATCHSIZE n [MINBATCHSIZE m]] [INPUTS name1 name2 ... OUTPUTS name1 name2 ...] BLOB model_blob
*/
int RedisAI_ModelSet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc < 4) return RedisModule_WrongArity(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  RedisModuleString* keystr;
  AC_GetRString(&ac, &keystr, 0);

  const char* bckstr;
  int backend;
  AC_GetString(&ac, &bckstr, NULL, 0); 
  if (strcasecmp(bckstr, "TF") == 0) {
    backend = RAI_BACKEND_TENSORFLOW;
  }
  else if (strcasecmp(bckstr, "TFLITE") == 0) {
    backend = RAI_BACKEND_TFLITE;
  }
  else if (strcasecmp(bckstr, "TORCH") == 0) {
    backend = RAI_BACKEND_TORCH;
  }
  else if (strcasecmp(bckstr, "ONNX") == 0) {
    backend = RAI_BACKEND_ONNXRUNTIME;
  }
  else {
    return RedisModule_ReplyWithError(ctx, "ERR unsupported backend");
  }

  const char* devicestr;
  AC_GetString(&ac, &devicestr, NULL, 0); 

  if (strlen(devicestr) > 10 ||
      strcasecmp(devicestr, "INPUTS") == 0 ||
      strcasecmp(devicestr, "OUTPUTS") == 0 ||
      strcasecmp(devicestr, "TAG") == 0 ||
      strcasecmp(devicestr, "BATCHSIZE") == 0 ||
      strcasecmp(devicestr, "MINBATCHSIZE") == 0 ||
      strcasecmp(devicestr, "BLOB") == 0
      ) {
    return RedisModule_ReplyWithError(ctx, "ERR Invalid DEVICE");
  }

  const char* tag = "";
  if (AC_AdvanceIfMatch(&ac, "TAG")) {
    AC_GetString(&ac, &tag, NULL, 0);
  }

  unsigned long long batchsize = 0;
  if (AC_AdvanceIfMatch(&ac, "BATCHSIZE")) {
    if (backend == RAI_BACKEND_TFLITE) {
      return RedisModule_ReplyWithError(ctx, "ERR Auto-batching not supported by the TFLITE backend");
    }
    if (AC_GetUnsignedLongLong(&ac, &batchsize, 0) != AC_OK) {
      return RedisModule_ReplyWithError(ctx, "ERR Invalid argument for BATCHSIZE");
    }
  }

  unsigned long long minbatchsize = 0;
  if (AC_AdvanceIfMatch(&ac, "MINBATCHSIZE")) {
    if (batchsize == 0) {
      return RedisModule_ReplyWithError(ctx, "ERR MINBATCHSIZE specified without BATCHSIZE");
    }
    if (AC_GetUnsignedLongLong(&ac, &minbatchsize, 0) != AC_OK) {
      return RedisModule_ReplyWithError(ctx, "ERR Invalid argument for MINBATCHSIZE");
    }
  }

  if (AC_IsAtEnd(&ac)) {
    return RedisModule_ReplyWithError(ctx, "ERR Insufficient arguments, missing model BLOB");
  }

  ArgsCursor optionsac;
  const char* blob_matches[] = {"BLOB"};
  AC_GetSliceUntilMatches(&ac, &optionsac, 1, blob_matches);

  if (optionsac.argc == 0 && backend == RAI_BACKEND_TENSORFLOW) {
    return RedisModule_ReplyWithError(ctx, "ERR Insufficient arguments, INPUTS and OUTPUTS not specified");
  }

  ArgsCursor inac = {0};
  ArgsCursor outac = {0};
  if (optionsac.argc > 0 && backend == RAI_BACKEND_TENSORFLOW) {
    if (!AC_AdvanceIfMatch(&optionsac, "INPUTS")) {
      return RedisModule_ReplyWithError(ctx, "ERR INPUTS not specified");
    }

    const char* matches[] = {"OUTPUTS"};
    AC_GetSliceUntilMatches(&optionsac, &inac, 1, matches);

    if (!AC_IsAtEnd(&optionsac)) {
      if (!AC_AdvanceIfMatch(&optionsac, "OUTPUTS")) {
        return RedisModule_ReplyWithError(ctx, "ERR OUTPUTS not specified");
      }

      AC_GetSliceToEnd(&optionsac, &outac);
    }
  }

  size_t ninputs = inac.argc;
  const char *inputs[ninputs];
  for (size_t i=0; i<ninputs; i++) {
    AC_GetString(&inac, inputs+i, NULL, 0); 
  }

  size_t noutputs = outac.argc;
  const char *outputs[noutputs];
  for (size_t i=0; i<noutputs; i++) {
    AC_GetString(&outac, outputs+i, NULL, 0); 
  }

  RAI_ModelOpts opts = {
    .batchsize = batchsize,
    .minbatchsize = minbatchsize,
    .backends_intra_op_parallelism = getBackendsIntraOpParallelism(),
    .backends_inter_op_parallelism = getBackendsInterOpParallelism(),
  };

  RAI_Model *model = NULL;

  AC_AdvanceUntilMatches(&ac, 1, blob_matches);

  if (AC_IsAtEnd(&ac)) {
    return RedisModule_ReplyWithError(ctx, "ERR Insufficient arguments, missing model BLOB");
  }

  AC_Advance(&ac);

  ArgsCursor blobsac;
  AC_GetSliceToEnd(&ac, &blobsac);

  size_t modellen;
  char *modeldef;

  if (blobsac.argc == 1) {
    AC_GetString(&blobsac, (const char**)&modeldef, &modellen, 0); 
  }
  else {
    const char *chunks[blobsac.argc];
    size_t chunklens[blobsac.argc];
    modellen = 0;
    while (!AC_IsAtEnd(&blobsac)) {
      AC_GetString(&blobsac, &chunks[blobsac.offset], &chunklens[blobsac.offset], 0); 
      modellen += chunklens[blobsac.offset-1];
    }
    
    modeldef = RedisModule_Calloc(modellen, sizeof(char));
    size_t offset = 0;
    for (size_t i=0; i<blobsac.argc; i++) {
      memcpy(modeldef + offset, chunks[i], chunklens[i]);
      offset += chunklens[i];
    }
  }

  RAI_Error err = {0};

  model = RAI_ModelCreate(backend, devicestr, tag, opts, ninputs, inputs, noutputs, outputs, modeldef, modellen, &err);

  if (err.code == RAI_EBACKENDNOTLOADED) {
    RedisModule_Log(ctx, "warning", "backend %s not loaded, will try loading default backend", bckstr);
    int ret = RAI_LoadDefaultBackend(ctx, backend);
    if (ret == REDISMODULE_ERR) {
      RedisModule_Log(ctx, "error", "could not load %s default backend", bckstr);
      int ret = RedisModule_ReplyWithError(ctx, "ERR Could not load backend");
      RAI_ClearError(&err);
      return ret;
    }
    RAI_ClearError(&err);
    model = RAI_ModelCreate(backend, devicestr, tag, opts, ninputs, inputs, noutputs, outputs, modeldef, modellen, &err);
  }

  if (blobsac.argc > 1) {
    RedisModule_Free(modeldef);
  }

  if (err.code != RAI_OK) {
    RedisModule_Log(ctx, "error", err.detail);
    int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
    RAI_ClearError(&err);
    return ret;
  }

  // TODO: if backend loaded, make sure there's a queue
  RunQueueInfo *run_queue_info = NULL;
  if (ensureRunQueue(devicestr,&run_queue_info) != REDISMODULE_OK){
    RAI_ModelFree(model, &err);
    if (err.code != RAI_OK) {
      RedisModule_Log(ctx, "error", err.detail);
      int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
      RAI_ClearError(&err);
      return ret;
    }
    return RedisModule_ReplyWithError(ctx, "ERR Could not initialize queue on requested device");
  }

  RedisModuleKey *key = RedisModule_OpenKey(ctx, keystr,
      REDISMODULE_READ|REDISMODULE_WRITE);
  int type = RedisModule_KeyType(key);
  if (type != REDISMODULE_KEYTYPE_EMPTY &&
      !(type == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_ModelType)) {
    RedisModule_CloseKey(key);
    RAI_ModelFree(model, &err);
    if (err.code != RAI_OK) {
      RedisModule_Log(ctx, "error", err.detail);
      int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
      RAI_ClearError(&err);
      return ret;
    }
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  RedisModule_ModuleTypeSetValue(key, RedisAI_ModelType, model);

  model->infokey = RAI_AddStatsEntry(ctx, keystr, RAI_MODEL, backend, devicestr, tag);

  RedisModule_CloseKey(key);

  RedisModule_ReplyWithSimpleString(ctx, "OK");

  RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}

/**
* AI.MODELGET model_key [META] [BLOB]
*/
int RedisAI_ModelGet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc < 2 || argc > 4) return RedisModule_WrongArity(ctx);

  RAI_Model *mto;
  RedisModuleKey *key;
  const int status = RAI_GetModelFromKeyspace( ctx, argv[1], &key, &mto, REDISMODULE_READ );
  if (status == REDISMODULE_ERR) {
    return REDISMODULE_ERR;
  }

  int meta = 0;
  int blob = 0;
  for (int i=2; i<argc; i++) {
    const char *optstr = RedisModule_StringPtrLen(argv[i], NULL);
    if (!strcasecmp(optstr, "META")) {
      meta = 1;
    }
    else if (!strcasecmp(optstr, "BLOB")) {
      blob = 1;
    }
  }

  if (!meta && !blob) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR no META or BLOB specified");
  }

  RAI_Error err = {0};

  char *buffer = NULL;
  size_t len = 0;

  if (blob) {
    RAI_ModelSerialize(mto, &buffer, &len, &err);

    if (err.code != RAI_OK) {
      #ifdef RAI_PRINT_BACKEND_ERRORS
      printf("ERR: %s\n", err.detail);
      #endif
      int ret = RedisModule_ReplyWithError(ctx, err.detail);
      RAI_ClearError(&err);
      if (*buffer) {
        RedisModule_Free(buffer);
      }
      return ret;
    }
  }

  if (!meta && blob) {
    RedisModule_ReplyWithStringBuffer(ctx, buffer, len);
    RedisModule_Free(buffer);
    RedisModule_CloseKey(key);
    return REDISMODULE_OK;
  }

  const int outentries = blob ? 16 : 14;

  RedisModule_ReplyWithArray(ctx, outentries);

  RedisModule_ReplyWithCString(ctx, "backend");
  const char *backendstr = RAI_BackendName(mto->backend);
  RedisModule_ReplyWithCString(ctx, backendstr);

  RedisModule_ReplyWithCString(ctx, "device");
  RedisModule_ReplyWithCString(ctx, mto->devicestr);

  RedisModule_ReplyWithCString(ctx, "tag");
  RedisModule_ReplyWithCString(ctx, mto->tag ? mto->tag : "");

  RedisModule_ReplyWithCString(ctx, "batchsize");
  RedisModule_ReplyWithLongLong(ctx, (long)mto->opts.batchsize);

  RedisModule_ReplyWithCString(ctx, "minbatchsize");
  RedisModule_ReplyWithLongLong(ctx, (long)mto->opts.minbatchsize);

  RedisModule_ReplyWithCString(ctx, "inputs");
  const size_t ninputs = array_len(mto->inputs);
  RedisModule_ReplyWithArray(ctx, (long)ninputs);

  for (size_t i = 0; i < ninputs; i++) {
    RedisModule_ReplyWithCString(ctx, mto->inputs[i]);
  }

  RedisModule_ReplyWithCString(ctx, "outputs");
  const size_t noutputs = array_len(mto->outputs);
  RedisModule_ReplyWithArray(ctx, (long)noutputs);

  for (size_t i = 0; i < noutputs; i++) {
    RedisModule_ReplyWithCString(ctx, mto->outputs[i]);
  }

  if (meta && blob) {
    RedisModule_ReplyWithCString(ctx, "blob");
    RedisModule_ReplyWithStringBuffer(ctx, buffer, len);
    RedisModule_Free(buffer);
  }

  RedisModule_CloseKey(key);

  return REDISMODULE_OK;
}

/**
* AI.MODELDEL model_key
*/
int RedisAI_ModelDel_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 2) return RedisModule_WrongArity(ctx);

  RAI_Model *mto;
  RedisModuleKey *key;
  const int status = RAI_GetModelFromKeyspace(ctx, argv[1], &key, &mto, REDISMODULE_READ|REDISMODULE_WRITE);
  if(status==REDISMODULE_ERR){
      return REDISMODULE_ERR;
  }

  RedisModule_DeleteKey(key);
  RedisModule_CloseKey(key);
  RedisModule_ReplicateVerbatim(ctx);

  return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

/** 
* AI._MODELSCAN
*/
int RedisAI_ModelScan_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 1) return RedisModule_WrongArity(ctx);

  RedisModule_Log(ctx, "warning", "MODELSCAN is experimental and might be removed in future versions");

  long long nkeys;
  RedisModuleString** keys;
  const char** tags;
  RAI_ListStatsEntries(RAI_MODEL, &nkeys, &keys, &tags);

  RedisModule_ReplyWithArray(ctx, nkeys);

  for (long long i=0; i<nkeys; i++) {
    RedisModule_ReplyWithArray(ctx, 2);
    RedisModule_ReplyWithString(ctx, keys[i]);
    RedisModule_ReplyWithCString(ctx, tags[i]);
  }

  RedisModule_Free(keys);
  RedisModule_Free(tags);

  return REDISMODULE_OK;
}

/**
 * AI.MODELRUN model_key INPUTS input_key1 ... OUTPUTS output_key1 ...
 *
 * The request is queued and evaded asynchronously from a separate thread. The
 * client blocks until the computation finishes.
 */
int RedisAI_ModelRun_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv,
                                  int argc) {
  if (argc < 3) return RedisModule_WrongArity(ctx);

  RedisAI_RunInfo *rinfo = NULL;
  if (RAI_InitRunInfo(&rinfo) == REDISMODULE_ERR) {
    return RedisModule_ReplyWithError(ctx, "ERR Unable to allocate the memory and initialise the RedisAI_RunInfo structure");
  }

  RAI_Model *mto;
  RedisModuleKey *modelKey;
  const int status = RAI_GetModelFromKeyspace(ctx, argv[1], &modelKey, &mto, REDISMODULE_READ);
  if(status==REDISMODULE_ERR){
      return REDISMODULE_ERR;
  }
  
  RedisModule_RetainString(NULL, argv[1]);
  rinfo->runkey = argv[1];
  rinfo->mctx = RAI_ModelRunCtxCreate(mto);

  const int parse_result = RedisAI_Parse_ModelRun_RedisCommand(ctx, argv,
                                   argc, &(rinfo->mctx), &(rinfo->outkeys), &mto, 0, NULL, 0, NULL, NULL);
  RedisModule_CloseKey(modelKey);
  // if the number of parsed args is negative something went wrong
  if(parse_result<0){
    return REDISMODULE_ERR;
  }

  RunQueueInfo *run_queue_info = NULL;
    // If the queue does not exist, initialize it
  if (ensureRunQueue(mto->devicestr,&run_queue_info) == REDISMODULE_ERR) {
    return RedisModule_ReplyWithError(ctx, "ERR Queue not initialized for device");
  }

  rinfo->client = RedisModule_BlockClient(ctx, RAI_ModelRunScriptRunReply, NULL, RedisAI_FreeData, 0);
  // RedisModule_SetDisconnectCallback(rinfo->client, RedisAI_Disconnected);

  pthread_mutex_lock(&run_queue_info->run_queue_mutex);
  queuePush(run_queue_info->run_queue, rinfo);
  pthread_cond_signal(&run_queue_info->queue_condition_var);
  pthread_mutex_unlock(&run_queue_info->run_queue_mutex);

  return REDISMODULE_OK;
}

/** 
* AI.SCRIPTRUN script_key fn_name INPUTS input_key1 ... OUTPUTS output_key1 ...
*/
int RedisAI_ScriptRun_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc < 4) return RedisModule_WrongArity(ctx);

  RedisAI_RunInfo *rinfo = NULL;
  if (RAI_InitRunInfo(&rinfo) == REDISMODULE_ERR) {
    return RedisModule_ReplyWithError(ctx, "ERR Unable to allocate the memory and initialise the RedisAI_RunInfo structure");
  }

  if (RedisModule_IsKeysPositionRequest(ctx)) {
    RedisModule_KeyAtPos(ctx, 1);
    for (int i=3; i<argc; i++) {
      const char* arg = RedisModule_StringPtrLen(argv[i], NULL);
      if (strcasecmp(arg, "INPUTS") == 0 || strcasecmp(arg, "OUTPUTS") == 0) {
        continue;
      }
      RedisModule_KeyAtPos(ctx, i);
    }
    return REDISMODULE_OK;
  }

  RedisModule_AutoMemory(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  RedisModuleString* keystr;
  AC_GetRString(&ac, &keystr, 0);

  RAI_Script *sto;
  RedisModuleKey *key;
  const int status = RAI_GetScriptFromKeyspace(ctx, argv[1], &key, &sto, REDISMODULE_READ);
  if(status==REDISMODULE_ERR){
      return REDISMODULE_ERR;
  }

  const char* fnname;
  AC_GetString(&ac, &fnname, NULL, 0); 

  ArgsCursor inac = {0};
  ArgsCursor outac = {0};

  if (!AC_AdvanceIfMatch(&ac, "INPUTS")) {
    return RedisModule_ReplyWithError(ctx, "ERR Insufficient arguments, INPUTS not specified");
  }

  const char* matches[] = {"OUTPUTS"};
  AC_GetSliceUntilMatches(&ac, &inac, 1, matches);

  if (!AC_AdvanceIfMatch(&ac, "OUTPUTS")) {
    return RedisModule_ReplyWithError(ctx, "ERR Insufficient arguments, OUTPUTS not specified");
  }

  AC_GetSliceToEnd(&ac, &outac);

  size_t ninputs = inac.argc;
  RedisModuleString *inputs[ninputs];
  for (size_t i=0; i<ninputs; i++) {
    AC_GetRString(&inac, inputs+i, 0); 
  }

  size_t noutputs = outac.argc;
  RedisModuleString *outputs[noutputs];
  for (size_t i=0; i<noutputs; i++) {
    AC_GetRString(&outac, outputs+i, 0); 
  }

  rinfo->sctx = RAI_ScriptRunCtxCreate(sto, fnname);

  for (size_t i=0; i<ninputs; i++) {
    RAI_Tensor *t;
    RedisModuleKey *argkey;
    const int status = RAI_GetTensorFromKeyspace(ctx, inputs[i], &argkey, &t, REDISMODULE_READ);
    if(status==REDISMODULE_ERR){
         RedisModule_CloseKey(key);
         RAI_FreeRunInfo(ctx,rinfo);
        return REDISMODULE_ERR;
    }
    RedisModule_CloseKey(argkey);
    if (!RAI_ScriptRunCtxAddInput(rinfo->sctx, t)) {
      RAI_FreeRunInfo(ctx,rinfo);
      RedisModule_CloseKey(key);
      return RedisModule_ReplyWithError(ctx, "ERR Input key not found");
    }
  }

  for (size_t i=0; i<noutputs; i++) {
    if (!RAI_ScriptRunCtxAddOutput(rinfo->sctx)) {
      RAI_FreeRunInfo(ctx,rinfo);
      RedisModule_CloseKey(key);
      return RedisModule_ReplyWithError(ctx, "ERR Output key not found");
    }
    RedisModule_RetainString(ctx, outputs[i]);
    array_append(rinfo->outkeys,outputs[i]);
  }
  
  RedisModule_RetainString(ctx, keystr);
  rinfo->runkey = keystr;
  RunQueueInfo *run_queue_info = NULL;
    // If the queue does not exist, initialize it
  if (ensureRunQueue(sto->devicestr,&run_queue_info) == REDISMODULE_ERR) {
    RAI_FreeRunInfo(ctx,rinfo);
    return RedisModule_ReplyWithError(ctx, "ERR Queue not initialized for device");
  }

  rinfo->client = RedisModule_BlockClient(ctx, RAI_ModelRunScriptRunReply, NULL, RedisAI_FreeData, 0);
  // RedisModule_SetDisconnectCallback(rinfo->client, RedisAI_Disconnected);

  pthread_mutex_lock(&run_queue_info->run_queue_mutex);
  queuePush(run_queue_info->run_queue, rinfo);
  pthread_cond_signal(&run_queue_info->queue_condition_var);
  pthread_mutex_unlock(&run_queue_info->run_queue_mutex);

  RedisModule_ReplicateVerbatim(ctx);
  RedisModule_CloseKey(key);

  return REDISMODULE_OK;
}

/**
 * AI.SCRIPTGET script_key [META] [SOURCE]
 */
int RedisAI_ScriptGet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc < 2 || argc > 4) return RedisModule_WrongArity(ctx);

  RAI_Script *sto;
  RedisModuleKey *key;
  const int status = RAI_GetScriptFromKeyspace(ctx, argv[1], &key, &sto, REDISMODULE_READ);
  if(status==REDISMODULE_ERR){
      return REDISMODULE_ERR;
  }

  int meta = 0;
  int source = 0;
  for (int i=2; i<argc; i++) {
    const char *optstr = RedisModule_StringPtrLen(argv[i], NULL);
    if (!strcasecmp(optstr, "META")) {
      meta = 1;
    }
    else if (!strcasecmp(optstr, "SOURCE")) {
      source = 1;
    }
  }

  if (!meta && !source) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, "ERR no META or SOURCE specified");
  }

  if (!meta && source) {
    RedisModule_ReplyWithCString(ctx, sto->scriptdef);
    RedisModule_CloseKey(key);
    return REDISMODULE_OK;
  }

  int outentries = source ? 6 : 4;

  RedisModule_ReplyWithArray(ctx, outentries);
  RedisModule_ReplyWithCString(ctx, "device");
  RedisModule_ReplyWithCString(ctx, sto->devicestr);
  RedisModule_ReplyWithCString(ctx, "tag");
  RedisModule_ReplyWithCString(ctx, sto->tag);
  if (source) {
    RedisModule_ReplyWithCString(ctx, "source");
    RedisModule_ReplyWithCString(ctx, sto->scriptdef);
  }
  RedisModule_CloseKey(key);
  return REDISMODULE_OK;
}

/**
 * AI.SCRIPTDEL script_key
 */
int RedisAI_ScriptDel_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 2) return RedisModule_WrongArity(ctx);

  RAI_Script *sto;
  RedisModuleKey *key;
  const int status = RAI_GetScriptFromKeyspace(ctx, argv[1], &key, &sto, REDISMODULE_WRITE);
  if(status==REDISMODULE_ERR){
      return REDISMODULE_ERR;
  }

  RedisModule_DeleteKey(key);
  RedisModule_CloseKey(key);

  RedisModule_ReplicateVerbatim(ctx);
  
  return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

/** 
* AI.SCRIPTSET script_key device [TAG tag] SOURCE script_source
*/
int RedisAI_ScriptSet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  RedisModule_AutoMemory(ctx);

  if (argc != 5 && argc != 7) return RedisModule_WrongArity(ctx);

  ArgsCursor ac;
  ArgsCursor_InitRString(&ac, argv+1, argc-1);

  RedisModuleString* keystr;
  AC_GetRString(&ac, &keystr, 0);

  const char* devicestr;
  AC_GetString(&ac, &devicestr, NULL, 0); 

  const char* tag = "";
  if (AC_AdvanceIfMatch(&ac, "TAG")) {
    AC_GetString(&ac, &tag, NULL, 0);
  }

  if (AC_IsAtEnd(&ac)) {
    return RedisModule_ReplyWithError(ctx, "ERR Insufficient arguments, missing script SOURCE");
  }

  size_t scriptlen;
  const char *scriptdef = NULL;

  if (AC_AdvanceIfMatch(&ac, "SOURCE")) {
    AC_GetString(&ac, &scriptdef, &scriptlen, 0); 
  }

  if (scriptdef == NULL) {
    return RedisModule_ReplyWithError(ctx, "ERR Insufficient arguments, missing script SOURCE");
  }

  RAI_Script *script = NULL;

  RAI_Error err = {0};
  script = RAI_ScriptCreate(devicestr, tag, scriptdef, &err);

  if (err.code == RAI_EBACKENDNOTLOADED) {
    RedisModule_Log(ctx, "warning", "Backend TORCH not loaded, will try loading default backend");
    int ret = RAI_LoadDefaultBackend(ctx, RAI_BACKEND_TORCH);
    if (ret == REDISMODULE_ERR) {
      RedisModule_Log(ctx, "error", "Could not load TORCH default backend");
      int ret = RedisModule_ReplyWithError(ctx, "ERR Could not load backend");
      RAI_ClearError(&err);
      return ret;
    }
    RAI_ClearError(&err);
    script = RAI_ScriptCreate(devicestr, tag, scriptdef, &err);
  }

  if (err.code != RAI_OK){
    #ifdef RAI_PRINT_BACKEND_ERRORS
    printf("ERR: %s\n", err.detail);
    #endif
    int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
    RAI_ClearError(&err);
    return ret;
  }

  RunQueueInfo *run_queue_info = NULL;
  // If the queue does not exist, initialize it
  if (ensureRunQueue(devicestr,&run_queue_info) == REDISMODULE_ERR) {
    RAI_ScriptFree(script, &err);
    if (err.code != RAI_OK) {
      #ifdef RAI_PRINT_BACKEND_ERRORS
      printf("ERR: %s\n", err.detail);
      #endif
      int ret = RedisModule_ReplyWithError(ctx, err.detail_oneline);
      RAI_ClearError(&err);
      return ret;
    }
    return RedisModule_ReplyWithError(ctx, "ERR Could not initialize queue on requested device");
  }

  RedisModuleKey *key = RedisModule_OpenKey(ctx, keystr,
      REDISMODULE_READ|REDISMODULE_WRITE);
  int type = RedisModule_KeyType(key);
  if (type != REDISMODULE_KEYTYPE_EMPTY &&
      !(type == REDISMODULE_KEYTYPE_MODULE &&
        RedisModule_ModuleTypeGetType(key) == RedisAI_ScriptType)) {
    RedisModule_CloseKey(key);
    return RedisModule_ReplyWithError(ctx, REDISMODULE_ERRORMSG_WRONGTYPE);
  }

  RedisModule_ModuleTypeSetValue(key, RedisAI_ScriptType, script);

  script->infokey = RAI_AddStatsEntry(ctx, keystr, RAI_SCRIPT, RAI_BACKEND_TORCH, devicestr, tag);

  RedisModule_CloseKey(key);

  RedisModule_ReplyWithSimpleString(ctx, "OK");

  RedisModule_ReplicateVerbatim(ctx);

  return REDISMODULE_OK;
}

/** 
* AI._SCRIPTSCAN
*/
int RedisAI_ScriptScan_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 1) return RedisModule_WrongArity(ctx);

  RedisModule_Log(ctx, "warning", "SCRIPTSCAN is experimental and might be removed in future versions");

  long long nkeys;
  RedisModuleString** keys;
  const char** tags;
  RAI_ListStatsEntries(RAI_SCRIPT, &nkeys, &keys, &tags);

  RedisModule_ReplyWithArray(ctx, nkeys);

  for (long long i=0; i<nkeys; i++) {
    RedisModule_ReplyWithArray(ctx, 2);
    RedisModule_ReplyWithString(ctx, keys[i]);
    RedisModule_ReplyWithCString(ctx, tags[i]);
  }

  RedisModule_Free(keys);
  RedisModule_Free(tags);

  return REDISMODULE_OK;
}

/**
* AI.INFO <model_or_script_key> [RESETSTAT]
*/
int RedisAI_Info_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc != 2 && argc != 3) return RedisModule_WrongArity(ctx);
  const char *runkey = RedisModule_StringPtrLen(argv[1], NULL);
  struct RedisAI_RunStats *rstats = NULL;
  if (RAI_GetRunStats(runkey, &rstats) == REDISMODULE_ERR) {
    return RedisModule_ReplyWithError(ctx, "ERR cannot find run info for key");
  }

  if(argc==3){
    const char *subcommand = RedisModule_StringPtrLen(argv[2], NULL);
    if (!strcasecmp(subcommand, "RESETSTAT")) {
      RAI_ResetRunStats(rstats);
      RedisModule_ReplyWithSimpleString(ctx, "OK");
      return REDISMODULE_OK;
    }
  }

  RedisModule_ReplyWithArray(ctx, 18);

  RedisModule_ReplyWithCString(ctx, "key");
  RedisModule_ReplyWithString(ctx, rstats->key);
  RedisModule_ReplyWithCString(ctx, "type");
  if (rstats->type == 0) {
    RedisModule_ReplyWithCString(ctx, "MODEL");
  }
  else {
    RedisModule_ReplyWithCString(ctx, "SCRIPT");
  }
  RedisModule_ReplyWithCString(ctx, "backend");
  RedisModule_ReplyWithCString(ctx, RAI_BackendName(rstats->backend));
  RedisModule_ReplyWithCString(ctx, "device");
  RedisModule_ReplyWithCString(ctx, rstats->devicestr);
  RedisModule_ReplyWithCString(ctx, "tag");
  RedisModule_ReplyWithCString(ctx, rstats->tag);
  RedisModule_ReplyWithCString(ctx, "duration");
  RedisModule_ReplyWithLongLong(ctx, rstats->duration_us);
  RedisModule_ReplyWithCString(ctx, "samples");
  if (rstats->type == 0) {
    RedisModule_ReplyWithLongLong(ctx, rstats->samples);
  }
  else {
    RedisModule_ReplyWithLongLong(ctx, -1);
  }
  RedisModule_ReplyWithCString(ctx, "calls");
  RedisModule_ReplyWithLongLong(ctx, rstats->calls);
  RedisModule_ReplyWithCString(ctx, "errors");
  RedisModule_ReplyWithLongLong(ctx, rstats->nerrors);

  return REDISMODULE_OK;
}

/** 
* AI.CONFIG [BACKENDSPATH <default_location_of_backend_libraries> | LOADBACKEND <backend_identifier> <location_of_backend_library>]
*/
int RedisAI_Config_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
  if (argc < 2) return RedisModule_WrongArity(ctx);

  const char *subcommand = RedisModule_StringPtrLen(argv[1], NULL);
  if (!strcasecmp(subcommand, "LOADBACKEND")) {
    return RedisAI_Config_LoadBackend(ctx, argv + 1, argc - 1);
  }

  if (!strcasecmp(subcommand, "BACKENDSPATH")) {
    if (argc > 2) {
      return RedisAI_Config_BackendsPath(
          ctx, RedisModule_StringPtrLen(argv[2], NULL));
    } else {
      return RedisModule_ReplyWithError(
          ctx, "ERR BACKENDSPATH: missing path argument");
    }
  }

  return RedisModule_ReplyWithError(ctx, "ERR unsupported subcommand");
}

/**
 * AI.DAGRUN [LOAD <nkeys> key1 key2... ] [PERSIST <nkeys> key1 key2... ] |>
 * [COMMAND1] |> [COMMAND2] |> [COMMANDN]
 *
 * The request is queued and evaded asynchronously from a separate thread. The
 * client blocks until the computation finishes.
 */
int RedisAI_DagRun_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv,
                                int argc) {
  if (argc < 4) return RedisModule_WrongArity(ctx);

  RedisAI_RunInfo *rinfo = NULL;
  if (RAI_InitRunInfo(&rinfo) == REDISMODULE_ERR) {
    return RedisModule_ReplyWithError(ctx, "ERR Unable to allocate the memory and initialise the RedisAI_RunInfo structure");
  }
  rinfo->use_local_context = 1;
  RAI_DagOp* currentDagOp = NULL;
  RAI_InitDagOp(&currentDagOp);
  array_append(rinfo->dagOps,currentDagOp);

  int persistFlag=0;
  int loadFlag=0;
  int chainingOpCount=0;
  const char* deviceStr = NULL;

  for (size_t argpos = 1; argpos <= argc - 1; argpos++) {
    const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
    if (!strcasecmp(arg_string, "LOAD")) {
      loadFlag=1;
      const int parse_result = RAI_parseDAGLoadArgs(
          ctx, &argv[argpos], argc - argpos,&(rinfo->dagTensorsLoadedContext), &(rinfo->dagTensorsContext), "|>");
      if (parse_result > 0) {
        argpos += parse_result - 1;
      } else {
        RAI_FreeRunInfo(ctx,rinfo);
        return REDISMODULE_ERR;
      }
    } else if (!strcasecmp(arg_string, "PERSIST")) {
      persistFlag = 1;
      const int parse_result =
          RAI_parseDAGPersistArgs(ctx, &argv[argpos], argc - argpos,
                                  &(rinfo->dagTensorsPersistentContext), "|>");
      if (parse_result > 0) {
        argpos += parse_result - 1;
      } else {
        RAI_FreeRunInfo(ctx, rinfo);
        return REDISMODULE_ERR;
      }
    } else if (!strcasecmp(arg_string, "|>")) {
      // on the first pipe operator, if LOAD or PERSIST were used, we've already
      // allocated memory
      if (!((persistFlag == 1 || loadFlag == 1) && chainingOpCount == 0)) {
        rinfo->dagNumberCommands++;
        RAI_DagOp *currentDagOp = NULL;
        RAI_InitDagOp(&currentDagOp);
        array_append(rinfo->dagOps, currentDagOp);
      }
      chainingOpCount++;
    } else {
      if (!strcasecmp(arg_string, "AI.TENSORGET")) {
        rinfo->dagOps[rinfo->dagNumberCommands]->commandType = REDISAI_DAG_CMD_TENSORGET;
      }
      if (!strcasecmp(arg_string, "AI.TENSORSET")) {
        rinfo->dagOps[rinfo->dagNumberCommands]->commandType = REDISAI_DAG_CMD_TENSORSET;
      }
      if (!strcasecmp(arg_string, "AI.MODELRUN")) {
        if (argc - 2 < argpos) {
          return RedisModule_WrongArity(ctx);
        }
        rinfo->dagOps[rinfo->dagNumberCommands]->commandType = REDISAI_DAG_CMD_MODELRUN;
        RAI_Model *mto;
        RedisModuleKey *modelKey;
        const int status = RAI_GetModelFromKeyspace(ctx, argv[argpos+1], &modelKey,
                                                    &mto, REDISMODULE_READ);
        if (status == REDISMODULE_ERR) {
          RAI_FreeRunInfo(ctx,rinfo);
          return REDISMODULE_ERR;
        } 
        if (deviceStr==NULL){
          deviceStr=mto->devicestr;
        }else{
          // If the device strings are not equivalent, reply with error ( for now )
          if(strcasecmp(mto->devicestr, deviceStr)!=0){            
            RAI_FreeRunInfo(ctx,rinfo);
            return RedisModule_ReplyWithError(ctx,"ERR multi-device DAGs not supported yet");;
          }
        }
        rinfo->dagOps[rinfo->dagNumberCommands]->runkey = argv[argpos+1];
        rinfo->dagOps[rinfo->dagNumberCommands]->mctx =
            RAI_ModelRunCtxCreate(mto);
      }
      RedisModule_RetainString(NULL, argv[argpos]);
      array_append(rinfo->dagOps[rinfo->dagNumberCommands]->argv, argv[argpos]);
      rinfo->dagOps[rinfo->dagNumberCommands]->argc++;
    }
  }

  RunQueueInfo *run_queue_info = NULL;
  // If there was no MODELRUN on the DAG, we default all ops to CPU
  if(deviceStr==NULL){
    deviceStr="CPU";
  }
  // If the queue does not exist, initialize it
  if (ensureRunQueue(deviceStr,&run_queue_info) == REDISMODULE_ERR) {
    RAI_FreeRunInfo(ctx,rinfo);
    return RedisModule_ReplyWithError(
        ctx, "ERR Queue not initialized for device");
  }

  rinfo->client = RedisModule_BlockClient(ctx, RedisAI_DagRun_Reply, NULL,
                                          NULL, 0);

  pthread_mutex_lock(&run_queue_info->run_queue_mutex);
  queuePush(run_queue_info->run_queue, rinfo);
  pthread_cond_signal(&run_queue_info->queue_condition_var);
  pthread_mutex_unlock(&run_queue_info->run_queue_mutex);

  return REDISMODULE_OK;
}

/**
 * AI.DAGRUN_RO [LOAD <nkeys> key1 key2... ] |> [COMMAND1] |> [COMMAND2] |> [COMMANDN]
 *
 * Read-only (no PERSIST) DAG execution.
 * The request is queued and evaded asynchronously from a separate thread. The
 * client blocks until the computation finishes.
 */
int RedisAI_DagRunRO_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv,
                                int argc) {
  if (argc < 4) return RedisModule_WrongArity(ctx);

  RedisAI_RunInfo *rinfo = NULL;
  if (RAI_InitRunInfo(&rinfo) == REDISMODULE_ERR) {
    return RedisModule_ReplyWithError(ctx, "ERR Unable to allocate the memory and initialise the RedisAI_RunInfo structure");
  }
  rinfo->use_local_context = 1;
  RAI_DagOp* currentDagOp = NULL;
  RAI_InitDagOp(&currentDagOp);
  array_append(rinfo->dagOps,currentDagOp);

  int loadFlag=0;
  int chainingOpCount=0;
  const char* deviceStr = NULL;

  for (size_t argpos = 1; argpos <= argc - 1; argpos++) {
    const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
    if (!strcasecmp(arg_string, "LOAD")) {
      loadFlag=1;
      const int parse_result = RAI_parseDAGLoadArgs(
          ctx, &argv[argpos], argc - argpos,&(rinfo->dagTensorsLoadedContext), &(rinfo->dagTensorsContext), "|>");
      if (parse_result > 0) {
        argpos += parse_result - 1;
      } else {
        RAI_FreeRunInfo(ctx,rinfo);
        return REDISMODULE_ERR;
      }
    } else if (!strcasecmp(arg_string, "PERSIST")) {
      return RedisModule_ReplyWithError(ctx,"ERR PERSIST cannot be specified in a read-only DAG");;
    } else if (!strcasecmp(arg_string, "|>")) {
      // on the first pipe operator, if LOAD or PERSIST were used, we've already
      // allocated memory
      if (!(loadFlag == 1 && chainingOpCount == 0)) {
        rinfo->dagNumberCommands++;
        RAI_DagOp *currentDagOp = NULL;
        RAI_InitDagOp(&currentDagOp);
        array_append(rinfo->dagOps, currentDagOp);
      }
      chainingOpCount++;
    } else {
      if (!strcasecmp(arg_string, "AI.TENSORGET")) {
        rinfo->dagOps[rinfo->dagNumberCommands]->commandType = REDISAI_DAG_CMD_TENSORGET;
      }
      if (!strcasecmp(arg_string, "AI.TENSORSET")) {
        rinfo->dagOps[rinfo->dagNumberCommands]->commandType = REDISAI_DAG_CMD_TENSORSET;
      }
      if (!strcasecmp(arg_string, "AI.MODELRUN")) {
        if (argc - 2 < argpos) {
          return RedisModule_WrongArity(ctx);
        }
        rinfo->dagOps[rinfo->dagNumberCommands]->commandType = REDISAI_DAG_CMD_MODELRUN;
        RAI_Model *mto;
        RedisModuleKey *modelKey;
        const int status = RAI_GetModelFromKeyspace(ctx, argv[argpos+1], &modelKey,
                                                    &mto, REDISMODULE_READ);
        if (status == REDISMODULE_ERR) {
          RAI_FreeRunInfo(ctx,rinfo);
          return REDISMODULE_ERR;
        } 
        if (deviceStr==NULL){
          deviceStr=mto->devicestr;
        }else{
          // If the device strings are not equivalent, reply with error ( for now )
          if(strcasecmp(mto->devicestr, deviceStr)!=0){            
            RAI_FreeRunInfo(ctx,rinfo);
            return RedisModule_ReplyWithError(ctx,"ERR multi-device DAGs not supported yet");;
          }
        }
        rinfo->dagOps[rinfo->dagNumberCommands]->runkey = argv[argpos+1];
        rinfo->dagOps[rinfo->dagNumberCommands]->mctx =
            RAI_ModelRunCtxCreate(mto);
      }
      RedisModule_RetainString(NULL, argv[argpos]);
      array_append(rinfo->dagOps[rinfo->dagNumberCommands]->argv, argv[argpos]);
      rinfo->dagOps[rinfo->dagNumberCommands]->argc++;
    }
  }

  RunQueueInfo *run_queue_info = NULL;
  // If there was no MODELRUN on the DAG, we default all ops to CPU
  if(deviceStr==NULL){
    deviceStr="CPU";
  }
  // If the queue does not exist, initialize it
  if (ensureRunQueue(deviceStr,&run_queue_info) == REDISMODULE_ERR) {
    RAI_FreeRunInfo(ctx,rinfo);
    return RedisModule_ReplyWithError(
        ctx, "ERR Queue not initialized for device");
  }

  rinfo->client = RedisModule_BlockClient(ctx, RedisAI_DagRun_Reply, NULL,
                                          NULL, 0);

  pthread_mutex_lock(&run_queue_info->run_queue_mutex);
  queuePush(run_queue_info->run_queue, rinfo);
  pthread_cond_signal(&run_queue_info->queue_condition_var);
  pthread_mutex_unlock(&run_queue_info->run_queue_mutex);

  return REDISMODULE_OK;
}

#define EXECUTION_PLAN_FREE_MSG 100

#define REGISTER_API(name, ctx) \
  if (RedisModule_ExportSharedAPI) {\
    if (RedisModule_ExportSharedAPI(ctx, "RedisAI_" #name, RAI_ ## name) != REDISMODULE_OK) {\
      RedisModule_Log(ctx, "warning", "Could not register RedisAI_%s", #name);\
      return REDISMODULE_ERR;\
    }\
  }

static int RAI_GetLLAPIVersion(){
  return REDISAI_LLAPI_VERSION;
}

static int RedisAI_RegisterApi(RedisModuleCtx* ctx) {

  if (!RedisModule_ExportSharedAPI) {
    RedisModule_Log(ctx, "warning", "Redis version does not support SharedAPI; running without exposing C API to other modules");
  }

  REGISTER_API(GetLLAPIVersion, ctx);

  REGISTER_API(InitError, ctx);
  REGISTER_API(ClearError, ctx);
  REGISTER_API(FreeError, ctx);
  REGISTER_API(GetError, ctx);
  REGISTER_API(GetErrorOneLine, ctx);
  REGISTER_API(GetErrorCode, ctx);

  REGISTER_API(TensorCreate, ctx);
  REGISTER_API(TensorCreateByConcatenatingTensors, ctx);
  REGISTER_API(TensorCreateBySlicingTensor, ctx);
  REGISTER_API(TensorDataSize, ctx);
  REGISTER_API(TensorFree, ctx);
  REGISTER_API(TensorSetData, ctx);
  REGISTER_API(TensorSetValueFromLongLong, ctx);
  REGISTER_API(TensorSetValueFromDouble, ctx);
  REGISTER_API(TensorGetValueAsDouble, ctx);
  REGISTER_API(TensorGetValueAsLongLong, ctx);
  REGISTER_API(TensorGetShallowCopy, ctx);
  REGISTER_API(TensorNumDims, ctx);
  REGISTER_API(TensorDim, ctx);
  REGISTER_API(TensorByteSize, ctx);
  REGISTER_API(TensorData, ctx);

  REGISTER_API(ModelCreate, ctx);
  REGISTER_API(ModelFree, ctx);
  REGISTER_API(ModelRunCtxCreate, ctx);
  REGISTER_API(ModelRunCtxAddInput, ctx);
  REGISTER_API(ModelRunCtxAddOutput, ctx);
  REGISTER_API(ModelRunCtxNumOutputs, ctx);
  REGISTER_API(ModelRunCtxOutputTensor, ctx);
  REGISTER_API(ModelRunCtxFree, ctx);
  REGISTER_API(ModelRun, ctx);
  REGISTER_API(ModelSerialize, ctx);
  REGISTER_API(ModelGetShallowCopy, ctx);

  REGISTER_API(ScriptCreate, ctx);
  REGISTER_API(ScriptFree, ctx);
  REGISTER_API(ScriptRunCtxCreate, ctx);
  REGISTER_API(ScriptRunCtxAddInput, ctx);
  REGISTER_API(ScriptRunCtxAddOutput, ctx);
  REGISTER_API(ScriptRunCtxNumOutputs, ctx);
  REGISTER_API(ScriptRunCtxOutputTensor, ctx);
  REGISTER_API(ScriptRunCtxFree, ctx);
  REGISTER_API(ScriptRun, ctx);
  REGISTER_API(ScriptGetShallowCopy, ctx);

  return REDISMODULE_OK;
}

int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {

  if (RedisModule_Init(ctx, "ai", RAI_ENC_VER, REDISMODULE_APIVER_1)
      == REDISMODULE_ERR) return REDISMODULE_ERR;

  getRedisVersion();
  RedisModule_Log(ctx, "notice", "Redis version found by RedisAI: %d.%d.%d - %s",
                      redisMajorVersion, redisMinorVersion, redisPatchVersion,
                      IsEnterprise() ? "enterprise" : "oss");
  if (IsEnterprise()) {
      RedisModule_Log(ctx, "notice", "Redis Enterprise version found by RedisAI: %d.%d.%d-%d",
                      rlecMajorVersion, rlecMinorVersion, rlecPatchVersion, rlecBuild);
  }

  if (redisMajorVersion < 5 ||
      (redisMajorVersion == 5 && redisMinorVersion == 0 && redisPatchVersion < 7)) {
    RedisModule_Log(ctx, "warning", "RedisAI requires Redis version equal or greater than 5.0.7");
    return REDISMODULE_ERR;
  }

  RedisModule_Log(ctx, "notice", "RedisAI version %d, git_sha=%s",
                  RAI_ENC_VER, REDISAI_GIT_SHA);
 
  int flags = RedisModule_GetContextFlags(ctx);

  if(RedisAI_RegisterApi(ctx) != REDISMODULE_OK){
    RedisModule_Log(ctx, "warning", "could not register RedisAI api\r\n");
    return REDISMODULE_ERR;
  }

  if(!RAI_TensorInit(ctx)){
    RedisModule_Log(ctx, "warning", "can not initialize tensor dt\r\n");
    return REDISMODULE_ERR;
  }

  if(!RAI_ModelInit(ctx)){
    RedisModule_Log(ctx, "warning", "can not initialize model dt\r\n");
    return REDISMODULE_ERR;
  }

  if(!RAI_ScriptInit(ctx)){
    RedisModule_Log(ctx, "warning", "can not initialize script dt\r\n");
    return REDISMODULE_ERR;
  }

  if (RedisModule_CreateCommand(ctx, "ai.tensorset", RedisAI_TensorSet_RedisCommand, "write deny-oom", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.tensorget", RedisAI_TensorGet_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.modelset", RedisAI_ModelSet_RedisCommand, "write deny-oom", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.modelget", RedisAI_ModelGet_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.modeldel", RedisAI_ModelDel_RedisCommand, "write", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.modelrun", RedisAI_ModelRun_RedisCommand, "write deny-oom getkeys-api", 3, 3, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai._modelscan", RedisAI_ModelScan_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.scriptset", RedisAI_ScriptSet_RedisCommand, "write deny-oom", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.scriptget", RedisAI_ScriptGet_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.scriptdel", RedisAI_ScriptDel_RedisCommand, "write", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.scriptrun", RedisAI_ScriptRun_RedisCommand, "write deny-oom getkeys-api", 4, 4, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai._scriptscan", RedisAI_ScriptScan_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.info", RedisAI_Info_RedisCommand, "readonly", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.config", RedisAI_Config_RedisCommand, "write", 1, 1, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.dagrun", RedisAI_DagRun_RedisCommand, "write deny-oom", 3, 3, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  if (RedisModule_CreateCommand(ctx, "ai.dagrun_ro", RedisAI_DagRunRO_RedisCommand, "readonly", 3, 3, 1)
      == REDISMODULE_ERR)
    return REDISMODULE_ERR;

  // Default configs
  RAI_BackendsPath = NULL;
  perqueueThreadPoolSize = REDISAI_DEFAULT_THREADS_PER_QUEUE;
  setBackendsInterOpParallelism(REDISAI_DEFAULT_INTER_OP_PARALLELISM);
  setBackendsIntraOpParallelism(REDISAI_DEFAULT_INTRA_OP_PARALLELISM);
  
  RAI_loadTimeConfig(ctx,argv,argc);

  run_queues = AI_dictCreate(&AI_dictTypeHeapStrings, NULL);
  RunQueueInfo *run_queue_info = NULL;
  if (ensureRunQueue("CPU",&run_queue_info) != REDISMODULE_OK){
    RedisModule_Log(ctx, "warning", "Queue not initialized for device CPU" );
    return REDISMODULE_ERR;
  }

  run_stats = AI_dictCreate(&AI_dictTypeHeapStrings, NULL);
  
  return REDISMODULE_OK;
}

extern AI_dictType AI_dictTypeHeapStrings;
