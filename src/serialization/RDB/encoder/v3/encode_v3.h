#pragma once
#include "../../../serialization_include.h"

void RAI_RDBSaveTensor_v3(RedisModuleIO *io, void *value);

void RAI_RDBSaveModel_v3(RedisModuleIO *io, void *value);

void RAI_RDBSaveScript_v3(RedisModuleIO *io, void *value);
