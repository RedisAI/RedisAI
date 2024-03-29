/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "../../serialization_include.h"

void *Decode_PreviousTensor(RedisModuleIO *rdb, int encver);

void *Decode_PreviousModel(RedisModuleIO *rdb, int encver);

void *Decode_PreviousScript(RedisModuleIO *rdb, int encver);