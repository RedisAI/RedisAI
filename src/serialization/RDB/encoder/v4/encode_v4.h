/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "../../../serialization_include.h"

void RAI_RDBSaveTensor_v4(RedisModuleIO *io, void *value);

void RAI_RDBSaveModel_v4(RedisModuleIO *io, void *value);

void RAI_RDBSaveScript_v4(RedisModuleIO *io, void *value);
