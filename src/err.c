#include "err.h"
#include "stdlib.h"
#include "assert.h"
#include "string.h"

#include "redismodule.h"

char *RAI_Chomp(const char *src) {
  char* str = RedisModule_Strdup(src);
  size_t len = strlen(src);
  for (size_t i=0; i<len; i++) {
    if (str[i] == '\n' || str[i] == '\r') {
      str[i] = ' ';
    }
  }
  return str;
}

void RAI_SetError(RAI_Error *err, RAI_ErrorCode code, const char *detail) {
  if (err->code != RAI_OK) {
    return;
  }
  assert(!err->detail);
  err->code = code;

  if (detail) {
    err->detail = RedisModule_Strdup(detail);
  } else {
    err->detail = RedisModule_Strdup("ERR Generic error");
  }

  err->detail_oneline = RAI_Chomp(err->detail);
}

void RAI_ClearError(RAI_Error *err) {
  if (err->detail) {
    RedisModule_Free(err->detail);
    err->detail = NULL;
  }
  if (err->detail_oneline) {
    RedisModule_Free(err->detail_oneline);
    err->detail_oneline = NULL;
  }
  err->code = RAI_OK;
}
