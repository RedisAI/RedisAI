/* Hash Tables Implementation.
 *
 * This file implements in-memory hash tables with insert/del/replace/find/
 * get-random-element operations. Hash tables will auto-resize if needed
 * tables of power of two in size are used, collisions are handled by
 * chaining. See the source code for more information... :)
 *
 * Copyright (c) 2006-2012, Salvatore Sanfilippo <antirez at gmail dot com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of Redis nor the names of its contributors may be used
 *     to endorse or promote products derived from this software without
 *     specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>
#include <stddef.h>

#ifndef __DICT_H
#define __DICT_H

#define DICT_OK  0
#define DICT_ERR 1

/* Unused arguments generate annoying warnings... */
#define DICT_NOTUSED(V) ((void)V)

typedef struct AI_dictEntry {
    void *key;
    union {
        void *val;
        uint64_t u64;
        int64_t s64;
        double d;
    } v;
    struct AI_dictEntry *next;
} AI_dictEntry;

typedef struct AI_dictType {
    uint64_t (*hashFunction)(const void *key);
    void *(*keyDup)(void *privdata, const void *key);
    void *(*valDup)(void *privdata, const void *obj);
    int (*keyCompare)(void *privdata, const void *key1, const void *key2);
    void (*keyDestructor)(void *privdata, void *key);
    void (*valDestructor)(void *privdata, void *obj);
} AI_dictType;

/* This is our hash table structure. Every dictionary has two of this as we
 * implement incremental rehashing, for the old to the new table. */
typedef struct AI_dictht {
    AI_dictEntry **table;
    unsigned long size;
    unsigned long sizemask;
    unsigned long used;
} AI_dictht;

typedef struct AI_dict {
    AI_dictType *type;
    void *privdata;
    AI_dictht ht[2];
    long rehashidx;          /* rehashing not in progress if rehashidx == -1 */
    unsigned long iterators; /* number of iterators currently running */
} AI_dict;

/* If safe is set to 1 this is a safe iterator, that means, you can call
 * dictAdd, dictFind, and other functions against the dictionary even while
 * iterating. Otherwise it is a non safe iterator, and only dictNext()
 * should be called while iterating. */
typedef struct AI_dictIterator {
    AI_dict *d;
    long index;
    int table, safe;
    AI_dictEntry *entry, *nextEntry;
    /* unsafe iterator fingerprint for misuse detection. */
    long long fingerprint;
} AI_dictIterator;

typedef void(AI_dictScanFunction)(void *privdata, const AI_dictEntry *de);
typedef void(AI_dictScanBucketFunction)(void *privdata, AI_dictEntry **bucketref);

/* This is the initial size of every hash table */
#define DICT_HT_INITIAL_SIZE 4

/* ------------------------------- Macros ------------------------------------*/
#define AI_dictFreeVal(d, entry)                                                                   \
    if ((d)->type->valDestructor)                                                                  \
    (d)->type->valDestructor((d)->privdata, (entry)->v.val)

#define AI_dictSetVal(d, entry, _val_)                                                             \
    do {                                                                                           \
        if ((d)->type->valDup)                                                                     \
            (entry)->v.val = (d)->type->valDup((d)->privdata, _val_);                              \
        else                                                                                       \
            (entry)->v.val = (_val_);                                                              \
    } while (0)

#define AI_dictSetSignedIntegerVal(entry, _val_)                                                   \
    do {                                                                                           \
        (entry)->v.s64 = _val_;                                                                    \
    } while (0)

#define AI_dictSetUnsignedIntegerVal(entry, _val_)                                                 \
    do {                                                                                           \
        (entry)->v.u64 = _val_;                                                                    \
    } while (0)

#define AI_dictSetDoubleVal(entry, _val_)                                                          \
    do {                                                                                           \
        (entry)->v.d = _val_;                                                                      \
    } while (0)

#define AI_dictFreeKey(d, entry)                                                                   \
    if ((d)->type->keyDestructor)                                                                  \
    (d)->type->keyDestructor((d)->privdata, (entry)->key)

#define AI_dictSetKey(d, entry, _key_)                                                             \
    do {                                                                                           \
        if ((d)->type->keyDup)                                                                     \
            (entry)->key = (d)->type->keyDup((d)->privdata, _key_);                                \
        else                                                                                       \
            (entry)->key = (_key_);                                                                \
    } while (0)

#define AI_dictCompareKeys(d, key1, key2)                                                          \
    (((d)->type->keyCompare) ? (d)->type->keyCompare((d)->privdata, key1, key2) : (key1) == (key2))

#define AI_dictHashKey(d, key)           (d)->type->hashFunction(key)
#define AI_dictGetKey(he)                ((he)->key)
#define AI_dictGetVal(he)                ((he)->v.val)
#define AI_dictGetSignedIntegerVal(he)   ((he)->v.s64)
#define AI_dictGetUnsignedIntegerVal(he) ((he)->v.u64)
#define AI_dictGetDoubleVal(he)          ((he)->v.d)
#define AI_dictSlots(d)                  ((d)->ht[0].size + (d)->ht[1].size)
#define AI_dictSize(d)                   ((d)->ht[0].used + (d)->ht[1].used)
#define AI_dictIsRehashing(d)            ((d)->rehashidx != -1)

/* API */
AI_dict *AI_dictCreate(AI_dictType *type, void *privDataPtr);
int AI_dictExpand(AI_dict *d, unsigned long size);
int AI_dictAdd(AI_dict *d, void *key, void *val);
AI_dictEntry *AI_dictAddRaw(AI_dict *d, void *key, AI_dictEntry **existing);
AI_dictEntry *AI_dictAddOrFind(AI_dict *d, void *key);
int AI_dictReplace(AI_dict *d, void *key, void *val);
int AI_dictDelete(AI_dict *d, const void *key);
AI_dictEntry *AI_dictUnlink(AI_dict *ht, const void *key);
void AI_dictFreeUnlinkedEntry(AI_dict *d, AI_dictEntry *he);
void AI_dictRelease(AI_dict *d);
AI_dictEntry *AI_dictFind(AI_dict *d, const void *key);
void *AI_dictFetchValue(AI_dict *d, const void *key);
int AI_dictResize(AI_dict *d);
AI_dictIterator *AI_dictGetIterator(AI_dict *d);
AI_dictIterator *AI_dictGetSafeIterator(AI_dict *d);
AI_dictEntry *AI_dictNext(AI_dictIterator *iter);
void AI_dictReleaseIterator(AI_dictIterator *iter);
AI_dictEntry *AI_dictGetRandomKey(AI_dict *d);
unsigned int AI_dictGetSomeKeys(AI_dict *d, AI_dictEntry **des, unsigned int count);
void AI_dictGetStats(char *buf, size_t bufsize, AI_dict *d);
uint64_t AI_dictGenHashFunction(const void *key, int len);
uint64_t AI_dictGenCaseHashFunction(const unsigned char *buf, int len);
void AI_dictEmpty(AI_dict *d, void(callback)(void *));
void AI_dictEnableResize(void);
void AI_dictDisableResize(void);
int AI_dictRehash(AI_dict *d, int n);
int AI_dictRehashMilliseconds(AI_dict *d, int ms);
void AI_dictSetHashFunctionSeed(uint8_t *seed);
uint8_t *AI_dictGetHashFunctionSeed(void);
unsigned long AI_dictScan(AI_dict *d, unsigned long v, AI_dictScanFunction *fn,
                          AI_dictScanBucketFunction *bucketfn, void *privdata);
uint64_t AI_dictGetHash(AI_dict *d, const void *key);
AI_dictEntry **AI_dictFindEntryRefByPtrAndHash(AI_dict *d, const void *oldptr, uint64_t hash);

#endif /* __DICT_H */
