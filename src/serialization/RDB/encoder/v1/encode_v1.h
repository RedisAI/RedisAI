#pragma once
#include "../../../serialization_include.h"

void RAI_RDBSaveTensor_v1(RedisModuleIO *io, void *value);

void RAI_RDBSaveModel_v1(RedisModuleIO *io, void *value);

void RAI_RDBSaveScript_v1(RedisModuleIO *io, void *value);
