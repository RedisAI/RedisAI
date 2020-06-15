#ifndef SRC_VERSION_H_
#define SRC_VERSION_H_

#define RAI_ENC_VER 999999

/* API versions. */
#define REDISAI_LLAPI_VERSION 1

/* This is the major / minor encver, to be used as
   argument to RM_CreateDataType, which expects
   encver to be < 1024 */
static const long long RAI_ENC_VER_MM = RAI_ENC_VER == 999999 ? 1023 : RAI_ENC_VER / 100;

#endif /* SRC_VERSION_H_ */
