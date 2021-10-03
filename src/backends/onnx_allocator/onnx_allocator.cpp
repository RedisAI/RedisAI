#include "onnx_allocator.h"
#include "../onnxruntime.h"
#include "onnxruntime_cxx_api.h"
#include <atomic>

struct RAIOrtAllocator : OrtAllocator {
    RAIOrtAllocator();
    ~RAIOrtAllocator();
    RAIOrtAllocator(const RAIOrtAllocator&) = delete;
    RAIOrtAllocator& operator=(const RAIOrtAllocator&) = delete;

    void* Alloc(size_t size);
    void Free(void* p);
    const OrtMemoryInfo* Info() const;
    unsigned long long NumAllocatorAccess() const;
    unsigned long long MemoryInUse() const;
    void SetMemoryLimit(unsigned long long max_memory);
    static RAIOrtAllocator *GetInstance();

private:
    std::atomic<unsigned long long> memory_inuse{0};
    std::atomic<unsigned long long> num_allocator_access{0};
    unsigned long long memory_limit = 0;
    OrtMemoryInfo* cpu_memory_info;
    static RAIOrtAllocator* allocator_instance;
};

RAIOrtAllocator* RAIOrtAllocator::allocator_instance = nullptr;

RAIOrtAllocator::RAIOrtAllocator() {
    OrtAllocator::version = ORT_API_VERSION;
    OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) { return static_cast<RAIOrtAllocator*>(this_)->Alloc(size); };
    OrtAllocator::Free = [](OrtAllocator* this_, void* p) { static_cast<RAIOrtAllocator*>(this_)->Free(p); };
    OrtAllocator::Info = [](const OrtAllocator* this_) { return static_cast<const RAIOrtAllocator*>(this_)->Info(); };
    Ort::ThrowOnError(Ort::GetApi().CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &cpu_memory_info));
    RAIOrtAllocator::allocator_instance = this;
}

RAIOrtAllocator::~RAIOrtAllocator() {
    Ort::GetApi().ReleaseMemoryInfo(cpu_memory_info);
}

void* RAIOrtAllocator::Alloc(size_t size) {
    // Allocate an additional 63 bytes to ensure that we can return an address which is
    // 64-byte aligned, and an additional space in the size of a pointer to store
    // the address that RedisModule_Alloc returns.
    int offset = 63 + sizeof(void *);
    void *allocated_address = (void *)RedisModule_Alloc(size + offset);
    size_t allocated_size = RedisModule_MallocSize(allocated_address);
    // Update the total number of bytes that onnx is using and the number of accesses
    // that onnx made to the allocator.
    size_t cur_memory = memory_inuse.load();
    if (memory_limit && cur_memory + allocated_size > memory_limit) {
        throw Ort::Exception("Onnxruntime memory limit exceeded, memory allocation failed.", ORT_RUNTIME_EXCEPTION);
    }
    memory_inuse.fetch_add(allocated_size);
    num_allocator_access.fetch_add(1);
    // This operation guarantees that p2 is the closest 64-aligned address to (p1+size_t).
    void **aligned_address = (void **)(((size_t)(allocated_address) + offset) & (~63));
    // This stores the address p1 right before p2 (so we can retrieve it when we free).
    aligned_address[-1] = allocated_address;
    return aligned_address;
}

void RAIOrtAllocator::Free(void* p) {
    if (p == nullptr) {
        return;
    }
    // Retrieve the address that we originally received from RedisModule_Alloc
    // (this is the address that we need to sent to RedisModule_Free).
    void *allocated_address = ((void **)p)[-1];
    size_t allocated_size = RedisModule_MallocSize(allocated_address);
    // Update the total number of bytes that onnx is using and the number of accesses
    // that onnx made to the allocator.
    memory_inuse.fetch_sub(allocated_size);
    num_allocator_access.fetch_add(1);
    RedisModule_Free(allocated_address);
}

const OrtMemoryInfo* RAIOrtAllocator::Info() const {
    return cpu_memory_info;
}

unsigned long long RAIOrtAllocator::NumAllocatorAccess() const {
    return num_allocator_access.load();
}

unsigned long long RAIOrtAllocator::MemoryInUse() const {
    return memory_inuse.load();
}

void RAIOrtAllocator::SetMemoryLimit(unsigned long long max_memory) {
    memory_limit = max_memory;
}

RAIOrtAllocator *RAIOrtAllocator::GetInstance() {
    return RAIOrtAllocator::allocator_instance;
}

OrtAllocator *CreateCustomAllocator(unsigned long long max_memory) {
    auto *allocator = new RAIOrtAllocator();
    allocator->SetMemoryLimit(max_memory);
    return allocator;
}

unsigned long long RAI_GetMemoryInfoORT() {
    return RAIOrtAllocator::GetInstance()->MemoryInUse();
}

unsigned long long RAI_GetMemoryAccessORT() {
    return RAIOrtAllocator::GetInstance()->NumAllocatorAccess();
}
