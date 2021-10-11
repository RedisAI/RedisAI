#pragma once
#include "serialization/serialization_include.h"

void *RAI_RDBLoadTensor_v4(RedisModuleIO *io);

void *RAI_RDBLoadModel_v4(RedisModuleIO *io);

void *RAI_RDBLoadScript_v4(RedisModuleIO *io);
