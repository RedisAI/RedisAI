/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "serialization/serialization_include.h"

void *RAI_RDBLoadTensor_v2(RedisModuleIO *io);

void *RAI_RDBLoadModel_v2(RedisModuleIO *io);

void *RAI_RDBLoadScript_v2(RedisModuleIO *io);
