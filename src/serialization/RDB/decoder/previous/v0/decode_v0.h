#pragma once
#include "../../../../serialization_include.h"

void *RAI_RDBLoadTensor_v0(RedisModuleIO *io);

void *RAI_RDBLoadModel_v0(RedisModuleIO *io);

void *RAI_RDBLoadScript_v0(RedisModuleIO *io);