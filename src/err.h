#ifndef SRC_ERR_H_
#define SRC_ERR_H_

typedef enum {
  RAI_OK = 0,
  RAI_EMODELIMPORT,
  RAI_EMODELCONFIGURE,
  RAI_EMODELCREATE,
  RAI_EMODELRUN,
  RAI_EMODELFREE,
  RAI_ESCRIPTIMPORT,
  RAI_ESCRIPTCONFIGURE,
  RAI_ESCRIPTCREATE,
  RAI_ESCRIPTRUN,
  RAI_EUNSUPPORTEDBACKEND,
  RAI_ESCRIPTFREE
} RAI_ErrorCode;

typedef struct RAI_Error {
  RAI_ErrorCode code;
  char* detail;
} RAI_Error;

RAI_Error RAI_InitError();

void RAI_SetError(RAI_Error *err, RAI_ErrorCode code, const char *detail);

void RAI_ClearError(RAI_Error *err);

#endif