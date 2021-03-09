#pragma once

#include "execution/run_info.h"

/**
 * @brief  Parse and validate DAGRUN command (Populate the rinfo object):
 * - parse LOAD, PERSIST, and TIMEOUT args. Persist is not allowed if the DAG is READ-ONLY (dag_to
 * is true).
 * - parse and validate every DAGop individually.
 * - Generate a unique key for every tensor name that appear in the DAG's ops.
 * (thus ensure that the operations will be done by the desired order).
 * @return Returns REDISMODULE_OK if the command is valid, REDISMODULE_ERR otherwise.
 */
int ParseDAGRunCommand(RedisAI_RunInfo *rinfo, RedisModuleCtx *ctx, RedisModuleString **argv,
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
int ParseDAGOps(RedisAI_RunInfo *rinfo, RAI_DagOp **ops);
