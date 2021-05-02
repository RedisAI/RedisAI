#include "dictionaries.h"
#include "string_utils.h"
#include "arr.h"

static array_t dict_arr_clone_fn(void *privdata, const void *arr) {
    array_t dest;
    array_clone(dest, (array_t)arr);
    return dest;
}

static void dict_arr_free_fn(void *privdata, void *arr) { array_free(arr); }

AI_dictType AI_dictTypeHeapStrings = {
    .hashFunction = RAI_StringsHashFunction,
    .keyDup = RAI_StringsKeyDup,
    .valDup = NULL,
    .keyCompare = RAI_StringsKeyCompare,
    .keyDestructor = RAI_StringsKeyDestructor,
    .valDestructor = NULL,
};

AI_dictType AI_dictTypeHeapRStrings = {
    .hashFunction = RAI_RStringsHashFunction,
    .keyDup = RAI_RStringsKeyDup,
    .valDup = NULL,
    .keyCompare = RAI_RStringsKeyCompare,
    .keyDestructor = RAI_RStringsKeyDestructor,
    .valDestructor = NULL,
};

AI_dictType AI_dictType_String_ArrSimple = {
    .hashFunction = RAI_StringsHashFunction,
    .keyDup = RAI_StringsKeyDup,
    .valDup = dict_arr_clone_fn,
    .keyCompare = RAI_StringsKeyCompare,
    .keyDestructor = RAI_StringsKeyDestructor,
    .valDestructor = dict_arr_free_fn,
};
