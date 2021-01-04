#pragma once
#include "../../serialization_include.h"

void *Decode_PreviousTensor(RedisModuleIO *rdb, int encver);

void *Decode_PreviousModel(RedisModuleIO *rdb, int encver);

void *Decode_PreviousScript(RedisModuleIO *rdb, int encver);