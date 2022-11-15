/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "execution/run_info.h"

/**
 * @brief  Parse and validate DAGEXECUTE command (Populate the rinfo object):
 * - parse KEYS, LOAD, PERSIST, and TIMEOUT args. Persist is not allowed if the DAG is READ-ONLY
 * (dag_to is true).
 * - parse and validate every DAGop individually. SCRIPTEXECUTE is not allowed if the DAG is
 * READ-ONLY.
 * - Generate a unique key for every tensor name that appear in the DAG's ops.
 * (thus ensure that the operations will be done by the desired order).
 * @return Returns REDISMODULE_OK if the command is valid, REDISMODULE_ERR otherwise.
 */
int ParseDAGExecuteCommand(RedisAI_RunInfo *rinfo, RedisModuleCtx *ctx, RedisModuleString **argv,
                           int argc, bool dag_ro);

/**
 * @brief Parse the arguments of the given ops in the DAGRUN command and build every op accordingly.
 * @param rinfo The DAG run info that will be populated with the ops if they are valid.
 * with its op,
 * @param ops A local array of ops, where every op has an argv field that points to an
 * array of RedisModule strings arguments, and an argc field which is the number of
 * args.
 * @return Returns REDISMODULE_OK if the command is valid, REDISMODULE_ERR otherwise.
 */
int ParseDAGExecuteOps(RedisAI_RunInfo *rinfo, RAI_DagOp **ops, bool ro);

int DAGInitialParsing(RedisAI_RunInfo *rinfo, RedisModuleCtx *ctx, RedisModuleString **argv,
                      int argc, bool dag_ro, RAI_DagOp ***dag_ops);
