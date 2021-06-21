#pragma once
#include "serialization/serialization_include.h"

void *RAI_RDBLoadTensor_v3(RedisModuleIO *io);

void *RAI_RDBLoadModel_v3(RedisModuleIO *io);

void *RAI_RDBLoadScript_v3(RedisModuleIO *io);
