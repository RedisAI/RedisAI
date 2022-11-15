/*
 *Copyright Redis Ltd. 2018 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */


/* SDS allocator selection.
 *
 * This file is used in order to change the SDS allocator at compile time.
 * Just define the following defines to what you want to use. Also add
 * the include of your alternate allocator if needed (not needed in order
 * to use the default libc allocator). */

#if defined(__MACH__) || defined(__FreeBSD__)
#include <stdlib.h>
#else
#include <malloc.h>
#endif
//#include "zmalloc.h"
#define s_malloc  malloc
#define s_realloc realloc
#define s_free    free
