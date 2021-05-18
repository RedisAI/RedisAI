#pragma once
#include "serialization/serialization_include.h"

void *RAI_RDBLoadTensor_v2(RedisModuleIO *io);

void *RAI_RDBLoadModel_v2(RedisModuleIO *io);

void *RAI_RDBLoadScript_v2(RedisModuleIO *io);