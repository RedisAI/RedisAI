#pragma once

#include "run_info.h"

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
 * @brief Parse the arguments of ops in the DAGRUN command and build (or extend) the DagOp object
 * accordingly.
 * @param rinfo The DAG run info with its op, where every op has an argv field that points to an
 * array of RedisModule strings the represents the op, and an argc field which is the number of
 * args.
 * @param first_op_ind The index of the first op in the for which we parse its argument and build
 * it.
 * @param num_ops The number of ops in the DAG the need to be parsed.
 * @return Returns REDISMODULE_OK if the command is valid, REDISMODULE_ERR otherwise.
 */
int ParseDAGOps(RedisAI_RunInfo *rinfo, size_t first_op_ind, size_t num_ops);
