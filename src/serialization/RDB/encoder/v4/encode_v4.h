#pragma once
#include "../../../serialization_include.h"

void RAI_RDBSaveTensor_v4(RedisModuleIO *io, void *value);

void RAI_RDBSaveModel_v4(RedisModuleIO *io, void *value);

void RAI_RDBSaveScript_v4(RedisModuleIO *io, void *value);
