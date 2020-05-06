/**
 * err.h
 *
 * Contains the structure and headers for a formal API to create, initialize,
 * get, reset, and free errors among different backends.
 */

#ifndef SRC_ERR_H_
#define SRC_ERR_H_

typedef enum {
  RAI_OK = 0,
  RAI_EMODELIMPORT,
  RAI_EMODELCONFIGURE,
  RAI_EMODELCREATE,
  RAI_EMODELRUN,
  RAI_EMODELSERIALIZE,
  RAI_EMODELFREE,
  RAI_ESCRIPTIMPORT,
  RAI_ESCRIPTCONFIGURE,
  RAI_ESCRIPTCREATE,
  RAI_ESCRIPTRUN,
  RAI_EUNSUPPORTEDBACKEND,
  RAI_EBACKENDNOTLOADED,
  RAI_ESCRIPTFREE,
  RAI_ETENSORSET,
  RAI_ETENSORGET,
  RAI_EDAGRUN,
} RAI_ErrorCode;

typedef struct RAI_Error {
  RAI_ErrorCode code;
  char *detail;
  char *detail_oneline;
} RAI_Error;

/**
 * Allocate the memory and initialise the RAI_Error.
 *
 * @param result Output parameter to capture allocated RAI_Error.
 * @return 0 on success, or 1 if the allocation
 * failed.
 */
int RAI_InitError(RAI_Error **err);

/**
 * Populates the RAI_Error data structure with the error details
 *
 * @param err
 * @param code
 * @param detail
 */
void RAI_SetError(RAI_Error *err, RAI_ErrorCode code, const char *detail);

/**
 * Return the error description
 *
 * @param err
 * @return error description
 * @param err
 */
const char* RAI_GetError(RAI_Error *err);

/**
 * Return the error description as one line
 *
 * @param err
 * @return error description as one line
 * @param err
 */
const char* RAI_GetErrorOneLine(RAI_Error *err);

/**
 * Return the error code
 *
 * @param err
 * @return error code
 * @param err
 */
RAI_ErrorCode RAI_GetErrorCode(RAI_Error *err);

/**
 * Resets an previously used/allocated RAI_Error
 *
 * @param err
 */
void RAI_ClearError(RAI_Error *err);

/**
 * Frees the memory of the RAI_Error
 *
 * @param err
 */
void RAI_FreeError(RAI_Error *err);

#endif
