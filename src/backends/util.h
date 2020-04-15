#ifndef SRC_BACKENDS_UTIL_H_
#define SRC_BACKENDS_UTIL_H_

#include <stdint.h>
#include <stdio.h>
#include <strings.h>

#include "config.h"

int parseDeviceStr(const char* devicestr, RAI_Device* device,
                   int64_t* deviceid);

/**
 * Get number of threads used for parallelism between independent operations, by
 * backend.
 * @return number of threads used for parallelism between independent
 * operations, by backend
 */
uint8_t getBackendsInterOpParallelism();

/**
 * Set number of threads used for parallelism between independent operations, by
 * backend.
 *
 * @param num_threads
 * @return 0 on success, or 1  if failed
 */
int setBackendsInterOpParallelism(uint8_t num_threads);

/**
 * Get number of threads used within an individual op for parallelism, by
 * backend.
 * @return number of threads used within an individual op for parallelism, by
 * backend.
 */
uint8_t getBackendsIntraOpParallelism();

/**
 * Set number of threads used within an individual op for parallelism, by
 * backend.
 *
 * @param num_threads
 * @return 0 on success, or 1  if failed
 */
int setBackendsIntraOpParallelism(uint8_t num_threads);

#endif /* SRC_BACKENDS_UTIL_H_ */
