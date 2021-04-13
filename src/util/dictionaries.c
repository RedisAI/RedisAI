#include "dictionaries.h"
#include "string_utils.h"
#include "arr.h"

AI_dictType AI_dictTypeHeapStringsVals = {
    .hashFunction = RAI_StringsHashFunction,
    .keyDup = RAI_StringsKeyDup,
    .valDup = NULL,
    .keyCompare = RAI_StringsKeyCompare,
    .keyDestructor = RAI_StringsKeyDestructor,
    .valDestructor = RAI_StringsKeyDestructor,
};

AI_dictType AI_dictTypeHeapStrings = {
    .hashFunction = RAI_StringsHashFunction,
    .keyDup = RAI_StringsKeyDup,
    .valDup = NULL,
    .keyCompare = RAI_StringsKeyCompare,
    .keyDestructor = RAI_StringsKeyDestructor,
    .valDestructor = NULL,
};

AI_dictType AI_dictTypeHeapRStringsVals = {
    .hashFunction = RAI_RStringsHashFunction,
    .keyDup = RAI_RStringsKeyDup,
    .valDup = NULL,
    .keyCompare = RAI_RStringsKeyCompare,
    .keyDestructor = RAI_RStringsKeyDestructor,
    .valDestructor = RAI_RStringsKeyDestructor,
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
    .valDup = array_clone_fn,
    .keyCompare = RAI_StringsKeyCompare,
    .keyDestructor = RAI_StringsKeyDestructor,
    .valDestructor = array_free,
};


AI_dictType AI_dictType_RString_ArrSimple = {
    .hashFunction = RAI_RStringsHashFunction,
    .keyDup = RAI_RStringsKeyDup,
    .valDup = array_clone_fn,
    .keyCompare = RAI_RStringsKeyCompare,
    .keyDestructor = RAI_RStringsKeyDestructor,
    .valDestructor = array_free,
};

