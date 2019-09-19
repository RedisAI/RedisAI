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

#define DICT_OK 0
#define DICT_ERR 1

/* Unused arguments generate annoying warnings... */
#define DICT_NOTUSED(V) ((void) V)

typedef struct Gears_dictEntry {
    void *key;
    union {
        void *val;
        uint64_t u64;
        int64_t s64;
        double d;
    } v;
    struct Gears_dictEntry *next;
} Gears_dictEntry;

typedef struct Gears_dictType {
    uint64_t (*hashFunction)(const void *key);
    void *(*keyDup)(void *privdata, const void *key);
    void *(*valDup)(void *privdata, const void *obj);
    int (*keyCompare)(void *privdata, const void *key1, const void *key2);
    void (*keyDestructor)(void *privdata, void *key);
    void (*valDestructor)(void *privdata, void *obj);
} Gears_dictType;

/* This is our hash table structure. Every dictionary has two of this as we
 * implement incremental rehashing, for the old to the new table. */
typedef struct Gears_dictht {
    Gears_dictEntry **table;
    unsigned long size;
    unsigned long sizemask;
    unsigned long used;
} Gears_dictht;

typedef struct Gears_dict {
    Gears_dictType *type;
    void *privdata;
    Gears_dictht ht[2];
    long rehashidx; /* rehashing not in progress if rehashidx == -1 */
    unsigned long iterators; /* number of iterators currently running */
} Gears_dict;

/* If safe is set to 1 this is a safe iterator, that means, you can call
 * dictAdd, dictFind, and other functions against the dictionary even while
 * iterating. Otherwise it is a non safe iterator, and only dictNext()
 * should be called while iterating. */
typedef struct Gears_dictIterator {
    Gears_dict *d;
    long index;
    int table, safe;
    Gears_dictEntry *entry, *nextEntry;
    /* unsafe iterator fingerprint for misuse detection. */
    long long fingerprint;
} Gears_dictIterator;

typedef void (Gears_dictScanFunction)(void *privdata, const Gears_dictEntry *de);
typedef void (Gears_dictScanBucketFunction)(void *privdata, Gears_dictEntry **bucketref);

/* This is the initial size of every hash table */
#define DICT_HT_INITIAL_SIZE     4

/* ------------------------------- Macros ------------------------------------*/
#define Gears_dictFreeVal(d, entry) \
    if ((d)->type->valDestructor) \
        (d)->type->valDestructor((d)->privdata, (entry)->v.val)

#define Gears_dictSetVal(d, entry, _val_) do { \
    if ((d)->type->valDup) \
        (entry)->v.val = (d)->type->valDup((d)->privdata, _val_); \
    else \
        (entry)->v.val = (_val_); \
} while(0)

#define Gears_dictSetSignedIntegerVal(entry, _val_) \
    do { (entry)->v.s64 = _val_; } while(0)

#define Gears_dictSetUnsignedIntegerVal(entry, _val_) \
    do { (entry)->v.u64 = _val_; } while(0)

#define Gears_dictSetDoubleVal(entry, _val_) \
    do { (entry)->v.d = _val_; } while(0)

#define Gears_dictFreeKey(d, entry) \
    if ((d)->type->keyDestructor) \
        (d)->type->keyDestructor((d)->privdata, (entry)->key)

#define Gears_dictSetKey(d, entry, _key_) do { \
    if ((d)->type->keyDup) \
        (entry)->key = (d)->type->keyDup((d)->privdata, _key_); \
    else \
        (entry)->key = (_key_); \
} while(0)

#define Gears_dictCompareKeys(d, key1, key2) \
    (((d)->type->keyCompare) ? \
        (d)->type->keyCompare((d)->privdata, key1, key2) : \
        (key1) == (key2))

#define Gears_dictHashKey(d, key) (d)->type->hashFunction(key)
#define Gears_dictGetKey(he) ((he)->key)
#define Gears_dictGetVal(he) ((he)->v.val)
#define Gears_dictGetSignedIntegerVal(he) ((he)->v.s64)
#define Gears_dictGetUnsignedIntegerVal(he) ((he)->v.u64)
#define Gears_dictGetDoubleVal(he) ((he)->v.d)
#define Gears_dictSlots(d) ((d)->ht[0].size+(d)->ht[1].size)
#define Gears_dictSize(d) ((d)->ht[0].used+(d)->ht[1].used)
#define Gears_dictIsRehashing(d) ((d)->rehashidx != -1)

/* API */
Gears_dict *Gears_dictCreate(Gears_dictType *type, void *privDataPtr);
int Gears_dictExpand(Gears_dict *d, unsigned long size);
int Gears_dictAdd(Gears_dict *d, void *key, void *val);
Gears_dictEntry *Gears_dictAddRaw(Gears_dict *d, void *key, Gears_dictEntry **existing);
Gears_dictEntry *Gears_dictAddOrFind(Gears_dict *d, void *key);
int Gears_dictReplace(Gears_dict *d, void *key, void *val);
int Gears_dictDelete(Gears_dict *d, const void *key);
Gears_dictEntry *Gears_dictUnlink(Gears_dict *ht, const void *key);
void Gears_dictFreeUnlinkedEntry(Gears_dict *d, Gears_dictEntry *he);
void Gears_dictRelease(Gears_dict *d);
Gears_dictEntry * Gears_dictFind(Gears_dict *d, const void *key);
void *Gears_dictFetchValue(Gears_dict *d, const void *key);
int Gears_dictResize(Gears_dict *d);
Gears_dictIterator *Gears_dictGetIterator(Gears_dict *d);
Gears_dictIterator *Gears_dictGetSafeIterator(Gears_dict *d);
Gears_dictEntry *Gears_dictNext(Gears_dictIterator *iter);
void Gears_dictReleaseIterator(Gears_dictIterator *iter);
Gears_dictEntry *Gears_dictGetRandomKey(Gears_dict *d);
unsigned int Gears_dictGetSomeKeys(Gears_dict *d, Gears_dictEntry **des, unsigned int count);
void Gears_dictGetStats(char *buf, size_t bufsize, Gears_dict *d);
uint64_t Gears_dictGenHashFunction(const void *key, int len);
uint64_t Gears_dictGenCaseHashFunction(const unsigned char *buf, int len);
void Gears_dictEmpty(Gears_dict *d, void(callback)(void*));
void Gears_dictEnableResize(void);
void Gears_dictDisableResize(void);
int Gears_dictRehash(Gears_dict *d, int n);
int Gears_dictRehashMilliseconds(Gears_dict *d, int ms);
void Gears_dictSetHashFunctionSeed(uint8_t *seed);
uint8_t *Gears_dictGetHashFunctionSeed(void);
unsigned long Gears_dictScan(Gears_dict *d, unsigned long v, Gears_dictScanFunction *fn, Gears_dictScanBucketFunction *bucketfn, void *privdata);
uint64_t Gears_dictGetHash(Gears_dict *d, const void *key);
Gears_dictEntry **Gears_dictFindEntryRefByPtrAndHash(Gears_dict *d, const void *oldptr, uint64_t hash);

extern Gears_dictType Gears_dictTypeHeapStrings;
extern Gears_dictType Gears_dictTypeHeapStringsVals;

#endif /* __DICT_H */
