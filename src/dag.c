#include "dag.h"

#include <pthread.h>
#include <stdbool.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "model.h"
#include "redisai.h"
#include "rmutil/alloc.h"
#include "rmutil/args.h"
#include "run_info.h"
#include "stats.h"
#include "tensor.h"
#include "util/arr_rm_alloc.h"
#include "util/dict.h"
#include "util/queue.h"

/**
 * Actual method running the DAGRUN Commands in the background
 * thread Called within `RedisAI_Run_ThreadMain`
 */
void *RedisAI_DagRunSession(RedisAI_RunInfo *rinfo) {
  for (size_t i = 0; i < array_len(rinfo->dagOps); i++) {
    RAI_DagOp *currentOp = rinfo->dagOps[i];
    switch (currentOp->commandType) {
      case REDISAI_DAG_CMD_TENSORSET: {
        RAI_Tensor *t = NULL;
        const int parse_result = RAI_parseTensorSetArgs(
            NULL, currentOp->argv, currentOp->argc, &t, 0, currentOp->err);
        if (parse_result > 0) {
          const char *key_string =
              RedisModule_StringPtrLen(currentOp->argv[1], NULL);
          const char *dictKey = RedisModule_Strdup(key_string);
          AI_dictReplace(rinfo->dagTensorsContext, dictKey, t);
          currentOp->result = REDISMODULE_OK;
        } else {
          currentOp->result = REDISMODULE_ERR;
        }
        break;
      }
      case REDISAI_DAG_CMD_TENSORGET: {
        const char *key_string =
            RedisModule_StringPtrLen(currentOp->argv[1], NULL);
        RAI_Tensor *t = NULL;
        currentOp->result = RAI_getTensorFromLocalContext(
            NULL, rinfo->dagTensorsContext, key_string, &t, currentOp->err);
        if (currentOp->result == REDISMODULE_OK) {
          RAI_Tensor *outTensor = NULL;
          // TODO: check tensor copy return value
          RAI_TensorCopyTensor(t, &outTensor);
          array_append(currentOp->outTensors, outTensor);
          currentOp->result = REDISMODULE_OK;
        }
        break;
      }
      case REDISAI_DAG_CMD_MODELRUN: {
        const int parse_result = RedisAI_Parse_ModelRun_RedisCommand(
            NULL, currentOp->argv, currentOp->argc, &(currentOp->mctx),
            &(currentOp->outkeys), &(currentOp->mctx->model), 1,
            &(rinfo->dagTensorsContext), 0, NULL, currentOp->err);

        if (parse_result > 0) {
          RAI_ModelRunCtx **mctxs = NULL;
          mctxs = array_new(RAI_ModelRunCtx *, 1);
          mctxs = array_append(mctxs, currentOp->mctx);
          currentOp->result = REDISMODULE_OK;
          const long long start = ustime();
          currentOp->result = RAI_ModelRun(mctxs, currentOp->err);
          currentOp->duration_us = ustime() - start;
          const size_t noutputs = RAI_ModelRunCtxNumOutputs(currentOp->mctx);
          for (size_t outputNumber = 0; outputNumber < noutputs;
               outputNumber++) {
            RAI_Tensor *tensor =
                RAI_ModelRunCtxOutputTensor(currentOp->mctx, outputNumber);
            if (tensor) {
              const char *key_string = RedisModule_StringPtrLen(
                  currentOp->outkeys[outputNumber], NULL);
              const char *dictKey = RedisModule_Strdup(key_string);
              AI_dictReplace(rinfo->dagTensorsContext, dictKey, tensor);
            } else {
              RAI_SetError(currentOp->err, RAI_EMODELRUN,
                           "ERR output tensor on DAG's MODELRUN was null");
              currentOp->result = REDISMODULE_ERR;
            }
          }
          array_free(mctxs);
        } else {
          currentOp->result = REDISMODULE_ERR;
        }
        break;
      }
      default: {
        /* unsupported DAG's command */
        RAI_SetError(currentOp->err, RAI_EDAGRUN,
                     "ERR unsupported command within DAG");
        currentOp->result = REDISMODULE_ERR;
        break;
      }
    }
  }
  if (rinfo->client != NULL) {
    RedisModule_UnblockClient(rinfo->client, rinfo);
  }
  return NULL;
}

int RedisAI_DagRun_Reply(RedisModuleCtx *ctx, RedisModuleString **argv,
                         int argc) {
  REDISMODULE_NOT_USED(argv);
  REDISMODULE_NOT_USED(argc);
  RedisAI_RunInfo *rinfo = RedisModule_GetBlockedClientPrivateData(ctx);
  RedisModule_ReplyWithArray(ctx, REDISMODULE_POSTPONED_ARRAY_LEN);
  for (size_t i = 0; i < array_len(rinfo->dagOps); i++) {
    RAI_DagOp *currentOp = rinfo->dagOps[i];
    switch (currentOp->commandType) {
      case REDISAI_DAG_CMD_TENSORSET: {
        rinfo->dagReplyLength++;
        if (currentOp->result == REDISMODULE_ERR) {
          RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
        } else {
          RedisModule_ReplyWithSimpleString(ctx, "OK");
        }
        break;
      }

      case REDISAI_DAG_CMD_TENSORGET: {
        rinfo->dagReplyLength++;
        if (currentOp->result == REDISMODULE_ERR) {
          RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
        } else {
          if (array_len(currentOp->outTensors) > 0) {
            RAI_Tensor *tensor = currentOp->outTensors[0];
            RAI_parseTensorGetArgs(ctx, currentOp->argv, currentOp->argc,
                                   tensor);
          } else {
            RedisModule_ReplyWithError(
                ctx, "ERR error getting tensor from local context");
          }
        }
        break;
      }

      case REDISAI_DAG_CMD_MODELRUN: {
        rinfo->dagReplyLength++;
        if (currentOp->result == REDISMODULE_ERR) {
          RedisModule_ReplyWithError(ctx, currentOp->err->detail_oneline);
        } else {
          RedisModule_ReplyWithSimpleString(ctx, "OK");
        }
        break;
      }

      default:
        /* no-op */
        break;
    }
  }

  AI_dictIterator *persist_iter =
      AI_dictGetSafeIterator(rinfo->dagTensorsPersistentContext);
  AI_dictEntry *persist_entry = AI_dictNext(persist_iter);
  while (persist_entry) {
    const char *persist_key_name = AI_dictGetKey(persist_entry);
    AI_dictEntry *tensor_entry =
        AI_dictFind(rinfo->dagTensorsContext, persist_key_name);
    if (tensor_entry) {
      RAI_Tensor *tensor = AI_dictGetVal(tensor_entry);
      RedisModuleKey *key;
      RedisModuleString *tensor_keyname = RedisModule_CreateString(
          ctx, persist_key_name, strlen(persist_key_name));
      const int status = RAI_OpenKey_Tensor(
          ctx, tensor_keyname, &key, REDISMODULE_READ | REDISMODULE_WRITE);
      if (status == REDISMODULE_ERR) {
        RedisModule_ReplyWithError(ctx, "ERR could not save tensor");
        rinfo->dagReplyLength++;
      } else {
        if (RedisModule_ModuleTypeSetValue(key, RedisAI_TensorType, tensor) !=
            REDISMODULE_OK) {
          RedisModule_ReplyWithError(ctx, "ERR could not save tensor");
          rinfo->dagReplyLength++;
        }
      }
      RedisModule_CloseKey(key);
      RedisAI_ReplicateTensorSet(ctx, tensor_keyname, tensor);
      // TODO: free Tensor
    } else {
      RedisModule_ReplyWithError(
          ctx, "ERR specified persistent key that was not used on DAG");
      rinfo->dagReplyLength++;

      RedisModule_Log(ctx, "warning",
                      "on DAGRUN's PERSIST pecified persistent key (%s) that "
                      "was not used on DAG. Logging all local context keys",
                      persist_key_name);
      AI_dictIterator *local_iter =
          AI_dictGetSafeIterator(rinfo->dagTensorsContext);
      AI_dictEntry *local_entry = AI_dictNext(local_iter);
      while (local_entry) {
        const char *localcontext_key_name = AI_dictGetKey(local_entry);
        RedisModule_Log(ctx, "warning", "DAG's local context key (%s)",
                        localcontext_key_name);
        local_entry = AI_dictNext(local_iter);
      }

      for (size_t opN = 0; opN < array_len(rinfo->dagOps); opN++) {
        RedisModule_Log(
            ctx, "warning", "DAG's op n#  %d - cmdType %d ( argc %d )", opN,
            rinfo->dagOps[opN]->commandType, rinfo->dagOps[opN]->argc);
      }
    }

    persist_entry = AI_dictNext(persist_iter);
  }
  AI_dictReleaseIterator(persist_iter);
  RedisModule_ReplySetArrayLength(ctx, rinfo->dagReplyLength);
  RAI_FreeRunInfo(ctx, rinfo);
  return REDISMODULE_OK;
}

/**
 * DAGRUN Building Block to parse [LOAD <nkeys> key1 key2... ]
 */
int RAI_parseDAGLoadArgs(RedisModuleCtx *ctx, RedisModuleString **argv,
                         int argc, AI_dict **loadedContextDict,
                         AI_dict **localContextDict,
                         const char *chaining_operator) {
  if (argc < 3) {
    RedisModule_WrongArity(ctx);
    return -1;
  }

  long long n_keys;
  const int retval = RedisModule_StringToLongLong(argv[1], &n_keys);
  if (retval != REDISMODULE_OK || n_keys <= 0) {
    RedisModule_ReplyWithError(
        ctx, "ERR invalid or negative value found in number of keys to LOAD");
    return -1;
  }

  int number_loaded_keys = 0;
  int separator_flag = 0;
  size_t argpos = 2;
  for (; (argpos <= argc - 1) && (number_loaded_keys < n_keys); argpos++) {
    const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
    if (!strcasecmp(arg_string, chaining_operator)) {
      separator_flag = 1;
      break;
    } else {
      RAI_Tensor *t;
      RedisModuleKey *key;
      const int status = RAI_GetTensorFromKeyspace(ctx, argv[argpos], &key, &t,
                                                   REDISMODULE_READ);
      if (status == REDISMODULE_ERR) {
        RedisModule_Log(
            ctx, "warning",
            "on DAGRUN's LOAD could not load tensor %s from keyspace",
            arg_string);
        return -1;
      }
      RedisModule_CloseKey(key);
      const char *dictKey = RedisModule_Strdup(arg_string);
      AI_dictAdd(*localContextDict, dictKey, t);
      const char *keyspacePersistKey = RedisModule_Strdup(dictKey);
      AI_dictAdd(*loadedContextDict, keyspacePersistKey, (void *)1);
      number_loaded_keys++;
    }
  }
  if (number_loaded_keys != n_keys) {
    RedisModule_WrongArity(ctx);
    return -1;
  }
  return argpos;
}

/**
 * DAGRUN Building Block to parse [PERSIST <nkeys> key1 key2... ]
 */
int RAI_parseDAGPersistArgs(RedisModuleCtx *ctx, RedisModuleString **argv,
                            int argc, AI_dict **persistContextDict,
                            const char *chaining_operator) {
  if (argc < 3) {
    RedisModule_WrongArity(ctx);
    return -1;
  }

  long long n_keys;
  const int retval = RedisModule_StringToLongLong(argv[1], &n_keys);
  if (retval != REDISMODULE_OK || n_keys <= 0) {
    RedisModule_ReplyWithError(
        ctx,
        "ERR invalid or negative value found in number of keys to PERSIST");
    return -1;
  }

  int number_loaded_keys = 0;
  int separator_flag = 0;
  size_t argpos = 2;
  for (; (argpos < argc) && (number_loaded_keys < n_keys); argpos++) {
    const char *arg_string = RedisModule_StringPtrLen(argv[argpos], NULL);
    if (!strcasecmp(arg_string, chaining_operator)) {
      separator_flag = 1;
      break;
    } else {
      const char *key = RedisModule_Strdup(arg_string);
      AI_dictAdd(*persistContextDict, key, (void *)1);
      number_loaded_keys++;
    }
  }
  if (number_loaded_keys != n_keys) {
    RedisModule_WrongArity(ctx);
    return -1;
  }
  return argpos;
}
