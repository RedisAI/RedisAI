

#ifndef SRC_RUN_INFO_H_
#define SRC_RUN_INFO_H_

#include "model.h"
#include "model_struct.h"
#include "redismodule.h"
#include "script.h"
#include "util/dict.h"

typedef struct RedisAI_RunInfo {
  RedisModuleBlockedClient *client;
  RedisModuleString *runkey;
  RedisModuleString **outkeys;
  RAI_ModelRunCtx *mctx;
  RAI_ScriptRunCtx *sctx;
  int status;
  long long duration_us;
  RAI_Error *err;
  int use_local_context;
  AI_dict *dagTensorsContext;
  AI_dict *dagTensorsPersistentContext;
  RedisModuleString ***dag_commands_argv;
  int *dag_commands_argc;
  int dag_number_commands;
  int dag_reply_length;
} RedisAI_RunInfo;

#endif /* SRC_RUN_INFO_H_ */
