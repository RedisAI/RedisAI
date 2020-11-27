#include "args.h"
#include "redismodule.h"
#include <float.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>

int AC_Advance(ArgsCursor *ac) { return AC_AdvanceBy(ac, 1); }

int AC_AdvanceBy(ArgsCursor *ac, size_t by) {
    if (ac->offset + by > ac->argc) {
        return AC_ERR_NOARG;
    } else {
        ac->offset += by;
    }
    return AC_OK;
}

int AC_AdvanceIfMatch(ArgsCursor *ac, const char *s) {
    const char *cur;
    if (AC_IsAtEnd(ac)) {
        return 0;
    }

    int rv = AC_GetString(ac, &cur, NULL, AC_F_NOADVANCE);
    assert(rv == AC_OK);
    rv = !strcasecmp(s, cur);
    if (rv) {
        AC_Advance(ac);
    }
    return rv;
}

#define MAYBE_ADVANCE()                                                                            \
    if (!(flags & AC_F_NOADVANCE)) {                                                               \
        AC_Advance(ac);                                                                            \
    }

static int tryReadAsDouble(ArgsCursor *ac, long long *ll, int flags) {
    double dTmp = 0.0;
    if (AC_GetDouble(ac, &dTmp, flags | AC_F_NOADVANCE) != AC_OK) {
        return AC_ERR_PARSE;
    }
    if (flags & AC_F_COALESCE) {
        *ll = dTmp;
        return AC_OK;
    }

    if ((double)(long long)dTmp != dTmp) {
        return AC_ERR_PARSE;
    } else {
        *ll = dTmp;
        return AC_OK;
    }
}

int AC_GetLongLong(ArgsCursor *ac, long long *ll, int flags) {
    if (ac->offset == ac->argc) {
        return AC_ERR_NOARG;
    }

    int hasErr = 0;
    // Try to parse the number as a normal integer first. If that fails, try
    // to parse it as a double. This will work if the number is in the format of
    // 3.00, OR if the number is in the format of 3.14 *AND* AC_F_COALESCE is set.
    if (ac->type == AC_TYPE_RSTRING) {
        if (RedisModule_StringToLongLong(AC_CURRENT(ac), ll) == REDISMODULE_ERR) {
            hasErr = 1;
        }
    } else {
        char *endptr = AC_CURRENT(ac);
        *ll = strtoll(AC_CURRENT(ac), &endptr, 10);
        if (*endptr != '\0' || *ll == LLONG_MIN || *ll == LLONG_MAX) {
            hasErr = 1;
        }
    }

    if (hasErr && tryReadAsDouble(ac, ll, flags) != AC_OK) {
        return AC_ERR_PARSE;
    }

    if ((flags & AC_F_GE0) && *ll < 0) {
        return AC_ERR_ELIMIT;
    }
    // Do validation
    if ((flags & AC_F_GE1) && *ll < 1) {
        return AC_ERR_ELIMIT;
    }
    MAYBE_ADVANCE();
    return AC_OK;
}

#define GEN_AC_FUNC(name, T, minVal, maxVal, isUnsigned)                                           \
    int name(ArgsCursor *ac, T *p, int flags) {                                                    \
        if (isUnsigned) {                                                                          \
            flags |= AC_F_GE0;                                                                     \
        }                                                                                          \
        long long ll;                                                                              \
        int rv = AC_GetLongLong(ac, &ll, flags | AC_F_NOADVANCE);                                  \
        if (rv) {                                                                                  \
            return rv;                                                                             \
        }                                                                                          \
        if (ll > maxVal || ll < minVal) {                                                          \
            return AC_ERR_ELIMIT;                                                                  \
        }                                                                                          \
        *p = ll;                                                                                   \
        MAYBE_ADVANCE();                                                                           \
        return AC_OK;                                                                              \
    }

GEN_AC_FUNC(AC_GetUnsignedLongLong, unsigned long long, 0, LLONG_MAX, 1)
GEN_AC_FUNC(AC_GetUnsigned, unsigned, 0, UINT_MAX, 1)
GEN_AC_FUNC(AC_GetInt, int, INT_MIN, INT_MAX, 0)
GEN_AC_FUNC(AC_GetU32, uint32_t, 0, UINT32_MAX, 1)
GEN_AC_FUNC(AC_GetU64, uint64_t, 0, UINT64_MAX, 1)

int AC_GetDouble(ArgsCursor *ac, double *d, int flags) {
    if (ac->type == AC_TYPE_RSTRING) {
        if (RedisModule_StringToDouble(ac->objs[ac->offset], d) != REDISMODULE_OK) {
            return AC_ERR_PARSE;
        }
    } else {
        char *endptr = AC_CURRENT(ac);
        *d = strtod(AC_CURRENT(ac), &endptr);
        if (*endptr != '\0' || *d == HUGE_VAL || *d == -HUGE_VAL) {
            return AC_ERR_PARSE;
        }
    }
    if ((flags & AC_F_GE0) && *d < 0.0) {
        return AC_ERR_ELIMIT;
    }
    if ((flags & AC_F_GE1) && *d < 1.0) {
        return AC_ERR_ELIMIT;
    }
    MAYBE_ADVANCE();
    return AC_OK;
}

int AC_GetRString(ArgsCursor *ac, RedisModuleString **s, int flags) {
    assert(ac->type == AC_TYPE_RSTRING);
    if (ac->offset == ac->argc) {
        return AC_ERR_NOARG;
    }
    *s = AC_CURRENT(ac);
    MAYBE_ADVANCE();
    return AC_OK;
}

int AC_GetString(ArgsCursor *ac, const char **s, size_t *n, int flags) {
    if (ac->offset == ac->argc) {
        return AC_ERR_NOARG;
    }
    if (ac->type == AC_TYPE_RSTRING) {
        *s = RedisModule_StringPtrLen(AC_CURRENT(ac), n);
    } else {
        *s = AC_CURRENT(ac);
        if (n) {
            if (ac->type == AC_TYPE_SDS) {
                *n = sdslen((const sds)*s);
            } else {
                *n = strlen(*s);
            }
        }
    }
    MAYBE_ADVANCE();
    return AC_OK;
}

const char *AC_GetStringNC(ArgsCursor *ac, size_t *len) {
    const char *s = NULL;
    if (AC_GetString(ac, &s, len, 0) != AC_OK) {
        return NULL;
    }
    return s;
}

int AC_GetVarArgs(ArgsCursor *ac, ArgsCursor *dst) {
    unsigned nargs;
    int rv = AC_GetUnsigned(ac, &nargs, 0);
    if (rv != AC_OK) {
        return rv;
    }
    return AC_GetSlice(ac, dst, nargs);
}

int AC_GetSlice(ArgsCursor *ac, ArgsCursor *dst, size_t n) {
    if (n > AC_NumRemaining(ac)) {
        return AC_ERR_NOARG;
    }

    dst->objs = ac->objs + ac->offset;
    dst->argc = n;
    dst->offset = 0;
    dst->type = ac->type;
    AC_AdvanceBy(ac, n);
    return 0;
}

int AC_AdvanceUntilMatches(ArgsCursor *ac, int n, const char **args) {
    const char *cur;
    if (AC_IsAtEnd(ac)) {
        return 0;
    }

    int rv;
    int matched = 0;
    while (!AC_IsAtEnd(ac)) {
        rv = AC_GetString(ac, &cur, NULL, AC_F_NOADVANCE);
        assert(rv == AC_OK);
        for (int i = 0; i < n; i++) {
            matched = !strcasecmp(args[i], cur);
            if (matched)
                break;
        }
        if (matched)
            break;
        AC_Advance(ac);
    }

    return rv;
}

int AC_GetSliceUntilMatches(ArgsCursor *ac, ArgsCursor *dest, int n, const char **args) {
    size_t offset0 = ac->offset;

    int rv = AC_AdvanceUntilMatches(ac, n, args);

    dest->objs = ac->objs + offset0;
    dest->argc = ac->offset - offset0;
    dest->offset = 0;
    dest->type = ac->type;

    return 0;
}

int AC_GetSliceToOffset(ArgsCursor *ac, ArgsCursor *dest, int offset) {
    size_t offset0 = ac->offset;

    if (offset0 > offset) {
        dest->objs = ac->objs + offset0;
        dest->argc = 0;
        dest->offset = 0;
        dest->type = ac->type;
        return 0;
    }

    size_t n = offset - offset0;

    dest->objs = ac->objs + offset0;
    dest->argc = n;
    dest->offset = 0;
    dest->type = ac->type;

    AC_AdvanceBy(ac, n);

    return 0;
}

int AC_GetSliceToEnd(ArgsCursor *ac, ArgsCursor *dest) {
    return AC_GetSliceToOffset(ac, dest, ac->argc);
}

static int parseSingleSpec(ArgsCursor *ac, ACArgSpec *spec) {
    switch (spec->type) {
    case AC_ARGTYPE_BOOLFLAG:
        *(int *)spec->target = 1;
        return AC_OK;
    case AC_ARGTYPE_BITFLAG:
        *(uint32_t *)(spec->target) |= spec->slicelen;
        return AC_OK;
    case AC_ARGTYPE_UNFLAG:
        *(uint32_t *)spec->target &= ~spec->slicelen;
        return AC_OK;
    case AC_ARGTYPE_DOUBLE:
        return AC_GetDouble(ac, spec->target, spec->intflags);
    case AC_ARGTYPE_INT:
        return AC_GetInt(ac, spec->target, spec->intflags);
    case AC_ARGTYPE_LLONG:
        return AC_GetLongLong(ac, spec->target, spec->intflags);
    case AC_ARGTYPE_ULLONG:
        return AC_GetUnsignedLongLong(ac, spec->target, spec->intflags);
    case AC_ARGTYPE_UINT:
        return AC_GetUnsigned(ac, spec->target, spec->intflags);
    case AC_ARGTYPE_STRING:
        return AC_GetString(ac, spec->target, spec->len, 0);
    case AC_ARGTYPE_RSTRING:
        return AC_GetRString(ac, spec->target, 0);
    case AC_ARGTYPE_SUBARGS:
        return AC_GetVarArgs(ac, spec->target);
    case AC_ARGTYPE_SUBARGS_N:
        return AC_GetSlice(ac, spec->target, spec->slicelen);
    default:
        fprintf(stderr, "Unknown type");
        abort();
    }
}

int AC_ParseArgSpec(ArgsCursor *ac, ACArgSpec *specs, ACArgSpec **errSpec) {
    const char *s = NULL;
    size_t n;
    int rv;

    if (errSpec) {
        *errSpec = NULL;
    }

    while (!AC_IsAtEnd(ac)) {
        if ((rv = AC_GetString(ac, &s, &n, AC_F_NOADVANCE) != AC_OK)) {
            return rv;
        }
        ACArgSpec *cur = specs;

        for (; cur->name != NULL; cur++) {
            if (n != strlen(cur->name)) {
                continue;
            }
            if (!strncasecmp(cur->name, s, n)) {
                break;
            }
        }

        if (cur->name == NULL) {
            return AC_ERR_ENOENT;
        }

        AC_Advance(ac);
        if ((rv = parseSingleSpec(ac, cur)) != AC_OK) {
            if (errSpec) {
                *errSpec = cur;
            }
            return rv;
        }
    }
    return AC_OK;
}
