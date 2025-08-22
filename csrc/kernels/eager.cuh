
#ifndef __EAGER_SUPPORT_H__
#define __EAGER_SUPPORT_H__

namespace deep_ep {

#include <stdint.h>

/*
 * @brief IB MTU based Tagging Layout
 * 
 * MTU-0    [0, ..., 4079, t0, t1, t2, ..., t15], total 4096 bytes
 * ...
 * MTU-last [0, ..., last, t0, t1, t2, ..., t15], total last + 1 + 16 bytes, note (last + 1) % 16 == 0
 */


// use IB MTU as segment size if PCIe Relaxed Ordering is Off, max IB MTU is 4096 bytes
#define PCIE_SEG_LEN_LOG (12) 
#define PCIE_SEG_LEN (1 << PCIE_SEG_LEN_LOG)
#define PCIE_SEG_LEN_MASK (PCIE_SEG_LEN - 1)  

#define AR_MSG_ALIGNMENT (1 << 4) 
// make TLP dst aligned by 256 bytes, avoid TLP spliting
#define AR_MSG_LONG_ALIGNMENT (1 << 8)

// for int4 alignment
#define PCIE_TAIL_SZ (1 << 4) 

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define CEIL_ALIGN(a, b) (CEIL_DIV(a, b) * (b))
#define NON_ZERO(x, y) ((x) < (y) ? (0) : ((x) - (y)))
#define PAGE_N(size) CEIL_DIV(size, PCIE_SEG_LEN - PCIE_TAIL_SZ)
#define FULL_MSG_LEN(size) ((size) + (PAGE_N(size) * PCIE_TAIL_SZ))
#define EXTEND_FOR_TAG_AND_ALIGN(size, alignment) CEIL_ALIGN(FULL_MSG_LEN(size), alignment)

#define DISPATCH_ROUND_INT 0x40000000
#define COMBINE_ROUND_INT 0xc0000000

#ifdef __CUDACC__

#include "utils.cuh"
__device__ __forceinline__ uint8_t ld_acquire_sys_global(const uint8_t *ptr) {
    uint32_t ret;
    asm volatile("ld.acquire.sys.global.u8 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret & 0xff;
}

__device__ __forceinline__ int ld_acquire_sys_global(int *ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int ld_acquire_shared(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.shared.cta.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ void st_release_shared(const int* ptr, int value) {
    asm volatile("st.release.shared.cta.s32 [%0], %1;" :: "l"(ptr), "r"(value) : "memory");
}

__device__  __forceinline__ void st_release_cta(const uint8_t *ptr, uint8_t val) {
    asm volatile("st.release.cta.u8 [%0], %1;"::"l"(ptr), "h"(static_cast<uint16_t>(val)) : "memory");
}

__device__  __forceinline__ void st_release_cta(const int4 *ptr, int4 val) {
    asm volatile("st.release.cta.v4.s32 [%0], {%1, %2, %3, %4};" 
            : : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
}

__device__  __forceinline__ void st_release_sys_global(const uint8_t *ptr, uint8_t val) {
    asm volatile("st.release.sys.global.u8 [%0], %1;"::"l"(ptr), "h"(static_cast<uint16_t>(val)) : "memory");
}

__device__  __forceinline__ void st_release_sys_global(const int4 *ptr, int4 val) {
    asm volatile("st.release.gpu.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};" 
            : : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) : "memory");
}

__device__ __forceinline__ void st_na_release(const int4 *ptr, int4 val) {
    asm volatile("st.release.gpu.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};" 
            : : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
}

__forceinline__ __device__ int warp_reduce_max(int value) {
    value = max(value, __shfl_xor_sync(0xffffffff, value, 16));
    value = max(value, __shfl_xor_sync(0xffffffff, value, 8));
    value = max(value, __shfl_xor_sync(0xffffffff, value, 4));
    value = max(value, __shfl_xor_sync(0xffffffff, value, 2));
    value = max(value, __shfl_xor_sync(0xffffffff, value, 1));
    return value;
}

__forceinline__ __device__ int warp_reduce_min(int value) {
    value = __reduce_min_sync(0xffffffff, value);
    return value;
}



#define TAG_V_OFFSET 0

/**
 * 
 * Phase 3: auto tagging
 */

#define SHIFTED_ADDR(a) (a + ((a) / (PCIE_SEG_LEN - PCIE_TAIL_SZ)) * PCIE_TAIL_SZ)
#define PTR_DIFF(a, b) (reinterpret_cast<uint8_t*>(a) - reinterpret_cast<uint8_t*>(b))
#define SHIFTED_ADDR_P(ptr, bptr) (ptr + ((PTR_DIFF(ptr, bptr) / (PCIE_SEG_LEN - PCIE_TAIL_SZ))) * PCIE_TAIL_SZ / sizeof(*(ptr)))
#define SHIFTED_ADDR_PS(ptr, bptr, smsglen) SHIFTED_ADDR_P(ptr + (PTR_DIFF(ptr, bptr) >= smsglen ? (sizeof(int4) / sizeof(*(ptr))) : 0), bptr)
#define IS_PAGE_SUB_HEAD(ptr, bptr, size) (((PTR_DIFF(ptr, bptr) == (size)) || (PTR_DIFF(ptr, bptr) % (PCIE_SEG_LEN - PCIE_TAIL_SZ)) == 0))
#define CHK_POSITION(bptr, size, pn, ptotal) (reinterpret_cast<uint8_t*>(bptr) + ((pn == ptotal - 1) ? SHIFTED_ADDR(size) : (((pn) << PCIE_SEG_LEN_LOG) + (PCIE_SEG_LEN - PCIE_TAIL_SZ))))
#define EXT_PAGE_N(size) CEIL_DIV(size, PCIE_SEG_LEN)

#define ZTAG(tag) (tag + TAG_V_OFFSET)
#define TAG_CNT_MASK 0x3fffffff
#define TAG_TYPE(tag) ((tag >> 31) & 1)
#define SHORT_TAG(tag) (((TAG_TYPE(tag) << 15) | ((((tag) & TAG_CNT_MASK) % 0x7fff) + 1)) << 16)

#define PARALLEL_SET_TAG(send_buf, len, tagv, exec_id, exec_total, st_func) {\
    const int __pages = PAGE_N(len);\
    for (int __pn = exec_id; __pn < __pages; __pn += exec_total) {\
        int *__check_ptr = reinterpret_cast<int*>(CHK_POSITION(send_buf, len, __pn, __pages));\
        /*printf("[rank %d]: dispatch round 0x%08x token %d st tag 0x%08x at offset %lu\n", rank, dispatch_round_n, token_idx, ZTAG(tagv), PTR_DIFF(__check_ptr, send_buf))*/;\
        /*EP_DEVICE_ASSERT(reinterpret_cast<uint64_t>(__check_ptr) % sizeof(int) == 0)*/;\
        st_func(__check_ptr, ZTAG(tagv));\
    }\
}

#define WARP_SET_TAIL_TAG(dst_buf, len, tagv) if constexpr (kEager != EAGER_OFF) {\
    __syncwarp();\
    /*EP_DEVICE_ASSERT(reinterpret_cast<uint64_t>(dst_buf + len) % sizeof(int) == 0)*/;\
    lane_id == 0 ? st_release_sys_global(reinterpret_cast<int*>(dst_buf + len), ZTAG(tagv)) : void(0);\
    /*if (lane_id == 0) printf("[rank %d]: dispatch round 0x%08x token %d st tail tag 0x%08x at offset %lu\n", rank, dispatch_round_n, token_idx, ZTAG(tagv), len)*/;\
}

#define WAIT_BIT(recv_buf, len, ext_len, exec_id, exec_total, tagv, intra_node) {\
    int __page_n = intra_node ? 1 : EXT_PAGE_N(ext_len);\
    for (int target = exec_id; target < __page_n; target += exec_total) {\
        int ld_value;\
        int* _check_ptr = intra_node ? reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(recv_buf) + ext_len) : reinterpret_cast<int*>(CHK_POSITION(recv_buf, len, target, __page_n));\
        /*EP_DEVICE_ASSERT(reinterpret_cast<uint64_t>(_check_ptr) % sizeof(int) == 0)*/;\
        while (true) {\
            ld_value = ld_acquire_sys_global(_check_ptr);\
            if (ld_value == ZTAG(tagv)) break;\
        }\
    }\
}

#define TRY_BIT(recv_buf, len, ext_len, exec_id, exec_total, tagv, ready, intra_node) {\
    int __page_n = intra_node ? 1 : EXT_PAGE_N(ext_len);\
    for (int target = exec_id; target < __page_n; target += exec_total) {\
        int ld_value;\
        int* _check_ptr = intra_node ? reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(recv_buf) + ext_len) : reinterpret_cast<int*>(CHK_POSITION(recv_buf, len, target, __page_n));\
        ld_value = ld_acquire_sys_global(_check_ptr);\
        if (ld_value == ZTAG(tagv)) {\
            ready = 0;\
            break;\
        }\
    }\
}

#define CHECK_TIME_MASK 0xfffff

#define WAIT_2BIT(recv_buf, len, ext_len, exec_id, exec_total, tagv, count_ptr, count_value, token_idx, intra_node) {\
    int __page_n = intra_node ? 1 : EXT_PAGE_N(ext_len);\
    for (int target = exec_id; target < __page_n; target += exec_total) {\
        int ld_value, w_cnt = 0;\
        int* _check_ptr = intra_node ? reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(recv_buf) + msg_distance - sizeof(int4)) : reinterpret_cast<int*>(CHK_POSITION(recv_buf, len, target, __page_n)) ;\
        while (true) {\
            ld_value = ld_acquire_sys_global(_check_ptr);\
            if (ld_value == ZTAG(tagv)) {\
                /*printf("[rank %d]: dispatch round 0x%08x expert %3d from rank %d get expected tag 0x%08x at offset: %lu\n", rank, dispatch_round_n, rank * num_local_experts + local_expert_idx, src_rank, ld_value, PTR_DIFF(_check_ptr, recv_buf))*/;\
                break;\
            }\
            if (count_value == 0) {\
                count_value = ld_acquire_sys_global(count_ptr);\
                count_value = ((count_value & 0xffff0000) == SHORT_TAG(tagv)) ? (count_value | 0xffff0000) : 0;\
                if (count_value != 0 && token_idx >= (-count_value-1)) {\
                    break;\
                }\
            }\
            w_cnt += 1;\
            if ((w_cnt & CHECK_TIME_MASK) == 0) printf("[rank %d]: dispatch round 0x%08x expert %3d from rank %d slot %d wait tag at offset %lu for %d times, 0x%08x != 0x%08x\n", rank, dispatch_round_n, rank * num_local_experts + local_expert_idx, src_rank, i, PTR_DIFF(_check_ptr, recv_buf), w_cnt, ld_value, ZTAG(tagv));\
        };\
    }\
}

#define TRY_2BIT(recv_buf, len, ext_len, exec_id, exec_total, tagv, count_ptr, count_value, ready, token_idx, intra_node) {\
    int __page_n = intra_node ? 1 : EXT_PAGE_N(ext_len);\
    if (exec_id == 0 && count_value == 0) {\
        count_value = ld_acquire_sys_global(count_ptr);\
        if ((count_value & 0xffff0000) == SHORT_TAG(tagv)) {\
            count_value = (count_value | 0xffff0000);\
        } else {\
            /*printf("[rank %d]: expert %d from %d, cnt tag mismatch, 0x%x != 0x%x\n", rank, global_expert_idx, src_rank, (count_value >> 16) & 0xffff, (tagv % 0xffff) + 1)*/;\
            count_value = 0;\
        }\
    }\
    __syncwarp();\
    for (int target = exec_id; target < __page_n; target += exec_total) {\
        int ld_value;\
        int* _check_ptr = intra_node ? reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(recv_buf) + ext_len) : reinterpret_cast<int*>(CHK_POSITION(recv_buf, len, target, __page_n));\
        ld_value = ld_acquire_sys_global(_check_ptr);\
        if (ld_value != ZTAG(tagv)) {\
            ready = 0;\
            break;\
        }\
    }\
    __syncwarp();\
}

#define NORMAL_ST(PTR, VALUE) *(PTR) = VALUE
#define NORMAL_LD(PTR) *(PTR)

#define LD_SHIFTED(LD_FUNC, SRC_PTR, SRC_BASE) LD_FUNC(SHIFTED_ADDR_P(SRC_PTR, SRC_BASE))

#define ST_SHIFTED(ST_FUNC, DST_PTR, DST_BASE, VALUE) ST_FUNC(SHIFTED_ADDR_P(DST_PTR, DST_BASE), VALUE)

#define N_LD_SHIFTED(SRC_PTR, SRC_BASE) LD_SHIFTED(NORMAL_LD, SRC_PTR, SRC_BASE)

#define N_ST_SHIFTED(DST_PTR, VALUE, DST_BASE) ST_SHIFTED(NORMAL_ST, DST_PTR, DST_BASE, VALUE)

#define LD_SHIFTED_S(LD_FUNC, SRC_PTR, SRC_BASE, SMSGLEN) LD_FUNC(SHIFTED_ADDR_PS(SRC_PTR, SRC_BASE, SMSGLEN))

#define ST_SHIFTED_S(ST_FUNC, DST_PTR, DST_BASE, SMSGLEN, VALUE) ST_FUNC(SHIFTED_ADDR_PS(DST_PTR, DST_BASE, SMSGLEN), VALUE)

#define N_LD_SHIFTED_S(SRC_PTR, SRC_BASE, SMSGLEN) LD_SHIFTED_S(NORMAL_LD, SRC_PTR, SRC_BASE, SMSGLEN)

#define N_ST_SHIFTED_S(DST_PTR, VALUE, DST_BASE, SMSGLEN) ST_SHIFTED_S(NORMAL_ST, DST_PTR, DST_BASE, SMSGLEN, VALUE)

#define UNROLLED_WARP_COPY_DST_AUTO_SHIFT(UNROLL_FACTOR, LANE_ID, N, DST, SRC, DST_RDMA_HEAD, LD_FUNC, ST_FUNC) \
{ \
    constexpr int kLoopStride = 32 * (UNROLL_FACTOR); \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)]; \
    auto __src = (SRC); \
    auto __dst = (DST); \
    for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            ST_SHIFTED(ST_FUNC, (__dst + __i + __j * 32), DST_RDMA_HEAD, unrolled_values[__j]);\
    } \
    for (int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); __i < (N); __i += 32) {\
        ST_SHIFTED(ST_FUNC, (__dst + __i), DST_RDMA_HEAD, LD_FUNC(__src + __i));\
    }\
}

#define UNROLLED_WARP_COPY_SRC_AUTO_SHIFT(UNROLL_FACTOR, LANE_ID, N, DST, SRC, SRC_RDMA_HEAD, SMSGLEN, LD_FUNC, ST_FUNC) \
{ \
    constexpr int kLoopStride = 32 * (UNROLL_FACTOR); \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)]; \
    auto __src = (SRC); \
    auto __dst = (DST); \
    for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) {\
            if constexpr (kUseFP8) {\
                unrolled_values[__j] = LD_SHIFTED(LD_FUNC, __src + __i + __j * 32, SRC_RDMA_HEAD); \
            } else {\
                unrolled_values[__j] = LD_SHIFTED_S(LD_FUNC, __src + __i + __j * 32, SRC_RDMA_HEAD, SMSGLEN); \
                /*printf("[rank %d]: dispatch round 0x%08x load data at offset %lu\n", rank, dispatch_round_n, PTR_DIFF(SHIFTED_ADDR_PS(__src + __i + __j * 32, SRC_RDMA_HEAD, SMSGLEN), SRC_RDMA_HEAD))*/;\
            }\
        }\
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]); \
    } \
    for (int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); __i < (N); __i += 32) {\
        if constexpr (kUseFP8) {\
            unrolled_values[0] = LD_SHIFTED(LD_FUNC, __src + __i, SRC_RDMA_HEAD);\
        } else {\
            unrolled_values[0] = LD_SHIFTED_S(LD_FUNC, __src + __i, SRC_RDMA_HEAD, SMSGLEN);\
            /*printf("[rank %d]: dispatch round 0x%08x load data at offset %lu\n", rank, dispatch_round_n, PTR_DIFF(SHIFTED_ADDR_PS(__src + __i, SRC_RDMA_HEAD, SMSGLEN), SRC_RDMA_HEAD))*/;\
        }\
        ST_FUNC(__dst + __i, unrolled_values[0]); \
    }\
}

#define INT_VALUE_NO_NAN(value) {\
    int __value = value;\
    nv_bfloat16 *__bf16_ptr = reinterpret_cast<nv_bfloat16*>(&__value);\
    EP_DEVICE_ASSERT(!isnan(static_cast<float>(__bf16_ptr[0])));\
    EP_DEVICE_ASSERT(!isnan(static_cast<float>(__bf16_ptr[1])));\
}

#define INT4_VALUE_NO_NAN(value) {\
    int4 __value = value;\
    nv_bfloat16 *__bf16_ptr = reinterpret_cast<nv_bfloat16*>(&__value);\
    for (int __u = 0; __u < 8; ++__u) {\
        EP_DEVICE_ASSERT(!isnan(static_cast<float>(__bf16_ptr[__u])));\
    }\
}


#define TMA_AUTO_TAG(tma_func, smem_ptr, gmem_ptr, bytes, gmem_base, tag_save, smsglen, tagv, kparam) {\
    if constexpr (kparam != EAGER_OFF) {\
        if (true) {\
            /*printf("[rank %d]: debug: modifying smem\n", rank)*/;\
            const auto diff = PTR_DIFF(gmem_ptr, gmem_base);\
            int __BASE_PN__ = diff >> PCIE_SEG_LEN_LOG;\
            int __TAIL_PN__ = (diff + bytes) >> PCIE_SEG_LEN_LOG;\
            if (__BASE_PN__ != __TAIL_PN__) {\
                int tag_tma_offset = (PCIE_SEG_LEN - PCIE_TAIL_SZ) - (diff & PCIE_SEG_LEN_MASK);\
                tag_save[__BASE_PN__] = *reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(smem_ptr) + tag_tma_offset);\
                /*printf("[rank %d]: exp %d send back rank %d token %d, insert middle tags, offset %lu, store 0x%08x at %d\n", rank, global_expert_idx, dst_rank, src_idx, tag_tma_offset + diff, tag_save[__BASE_PN__], __BASE_PN__)*/;\
                /*INT_VALUE_NO_NAN(tag_save[__BASE_PN__])*/;\
                st_release_cta(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(smem_ptr) + tag_tma_offset), ZTAG(tagv));\
            }\
            if (smsglen >= diff && smsglen < diff + bytes) {\
                const auto jump_tag_offset = (smsglen & PCIE_SEG_LEN_MASK) - (diff & PCIE_SEG_LEN_MASK);\
                tag_save[MAX_PAGES+1] = *reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(smem_ptr) + jump_tag_offset);\
                /*printf("[rank %d]: exp %d send back rank %d token %d, insert jump tags, offset %lu, store 0x%08x at %d\n", rank, global_expert_idx, dst_rank, src_idx, jump_tag_offset + diff, tag_save[MAX_PAGES+1], MAX_PAGES+1)*/;\
                st_release_cta(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(smem_ptr) + jump_tag_offset), 0);\
            }\
        }\
    }\
    tma_func(smem_ptr, gmem_ptr, bytes);\
}

#endif // __CUDACC__

#define EAGER_LOAD 0
#define EAGER_OFF 1
#define EAGER_CHK 2
#define EAGER_FULL 3
#define EAGER_DEBUG 10

#define SWITCH_EAGER(inner_macro, ...) \
do { \
    switch (eager_opt) { \
        case EAGER_LOAD: inner_macro(EAGER_LOAD, __VA_ARGS__); break; \
        case EAGER_OFF: inner_macro(EAGER_OFF, __VA_ARGS__); break; \
        case EAGER_CHK: inner_macro(EAGER_CHK, __VA_ARGS__); break; \
        case EAGER_FULL: inner_macro(EAGER_FULL, __VA_ARGS__); break; \
        default: EP_HOST_ASSERT(false && "Unsupported EAGER option"); \
    } \
} while (0); break;

};

#endif // __AR_SUPPORT_H__