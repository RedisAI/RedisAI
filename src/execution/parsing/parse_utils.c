#include "parse_utils.h"
#include "string.h"

int ParseTimeout(RedisModuleString *timeout_arg, RAI_Error *error, long long *timeout) {

    const int retval = RedisModule_StringToLongLong(timeout_arg, timeout);
    if (retval != REDISMODULE_OK || *timeout <= 0) {
        RAI_SetError(error, RAI_EMODELRUN, "ERR Invalid value for TIMEOUT");
        return REDISMODULE_ERR;
    }
    return REDISMODULE_OK;
}

const char *ScriptCommand_GetFunctionName(RedisModuleString *functionName) {
    const char *functionName_cstr = RedisModule_StringPtrLen(functionName, NULL);
    return functionName_cstr;
}
