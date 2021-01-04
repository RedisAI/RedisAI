#pragma once
#include "../serialization_include.h"

void RAI_AOFRewriteTensor(RedisModuleIO *aof, RedisModuleString *key, void *value);

void RAI_AOFRewriteModel(RedisModuleIO *aof, RedisModuleString *key, void *value);

void RAI_AOFRewriteScript(RedisModuleIO *aof, RedisModuleString *key, void *value);
