#pragma once
#include "serialization/serialization_include.h"

void *RAI_RDBLoadTensor_v1(RedisModuleIO *io);

void *RAI_RDBLoadModel_v1(RedisModuleIO *io);

void *RAI_RDBLoadScript_v1(RedisModuleIO *io);