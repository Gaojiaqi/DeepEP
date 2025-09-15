
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
#define ROUND_MASK 0x3fffffff

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

__device__ __forceinline__ int ld_relaxed_sys_global(int *ptr) {
    int ret;
    asm volatile("ld.relaxed.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
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


// x < 1048576 approximation
#define DIV255(x) (((x) * 0x8081U) >> 23)
#define DIV4080(x) DIV255((x) >> 4)


#define SHIFTED_ADDR(a) ((a) + (DIV4080(a) * PCIE_TAIL_SZ))
#define PTR_DIFF(a, b) (reinterpret_cast<uint8_t*>(a) - reinterpret_cast<uint8_t*>(b))
#define SHIFTED_ADDR_P(ptr, bptr) ((ptr) + DIV4080(PTR_DIFF(ptr, bptr)) * PCIE_TAIL_SZ / sizeof(*(ptr)))
#define IS_PAGE_SUB_HEAD(ptr, bptr, size) (((PTR_DIFF(ptr, bptr) == (size)) || (PTR_DIFF(ptr, bptr) % (PCIE_SEG_LEN - PCIE_TAIL_SZ)) == 0))
#define CHK_POSITION(bptr, ext_size, pn, ptotal) (reinterpret_cast<uint8_t*>(bptr) + ((pn == ptotal - 1) ? (ext_size - PCIE_TAIL_SZ) : (((pn) << PCIE_SEG_LEN_LOG) + (PCIE_SEG_LEN - PCIE_TAIL_SZ))))
#define EXT_PAGE_N(size) ((size >> PCIE_SEG_LEN_LOG) + ((size & PCIE_SEG_LEN_MASK) != 0))

#define TAG_V_OFFSET 0
#define ZTAG(tag) (tag + TAG_V_OFFSET)
#define TAG_CNT_MASK 0x3fffffff
#define TAG_TYPE(tag) ((tag >> 31) & 1)
#define SHORT_TAG(tag) (((TAG_TYPE(tag) << 15) | ((((tag) & TAG_CNT_MASK) % 0x7fff) + 1)) << 16)
#define CHECK_TIME_MASK 0xffffff
#define FINAL_TIME_MASK 0x10000000


#define PARALLEL_SET_TAG(send_buf, ext_len, tagv, exec_id, exec_total, st_func) {\
    const int __pages = EXT_PAGE_N(ext_len);\
    for (int __pn = exec_id; __pn < __pages; __pn += exec_total) {\
        int *__check_ptr = reinterpret_cast<int*>(CHK_POSITION(send_buf, ext_len, __pn, __pages));\
        st_func(__check_ptr, ZTAG(tagv));\
    }\
}

#define WAIT_BIT(recv_buf, ext_len, exec_id, exec_total, tagv, token_idx, topk_i, topk_idx, exp_rank, kernel_name) {\
    int __page_n = EXT_PAGE_N(ext_len);\
    for (int target = exec_id; target < __page_n; target += exec_total) {\
        int ld_value, w_cnt;\
        int* __check_ptr = reinterpret_cast<int*>(CHK_POSITION(recv_buf, ext_len, target, __page_n));\
        while (true) {\
            ld_value = ld_acquire_sys_global(__check_ptr);\
            if (ld_value == ZTAG(tagv)) break;\
            w_cnt += 1;\
            if (w_cnt == FINAL_TIME_MASK) {\
                printf("[rank %d]: [EAGER " kernel_name " HANG] round 0x%08x token %d topk %d from/to exp %d at rank %d, check offset %lu, %d times, 0x%08x != 0x%08x\n", rank, tagv, token_idx, topk_i, topk_idx, exp_rank, PTR_DIFF(__check_ptr, recv_buf), w_cnt, ld_value, ZTAG(tagv));\
                break;\
            }\
        }\
    }\
    __syncwarp();\
}

#define WARP_WAIT_LEN_AND_BIT(recv_buf, tagv, token_idx, topk_i, topk_idx, exp_rank, kernel_name) {\
    int head_value, data_len;\
    if (lane_id == 0) {\
        while (((head_value = ld_acquire_sys_global(reinterpret_cast<int*>(recv_buf))) & 0xffff0000u) != SHORT_TAG(tagv));\
        data_len = head_value & 0xffff;\
    }\
    data_len = __shfl_sync(0xffffffff, data_len, 0);\
    int ext_len = data_len + 2 * sizeof(int);\
    WAIT_BIT(recv_buf, ext_len, lane_id, 32, tagv, token_idx, topk_i, topk_idx, exp_rank, kernel_name);\
}

#define WAIT_2BIT(recv_buf, ext_len, exec_id, exec_total, tagv, count_ptr, count_value, token_idx, warp_id, count_cache_ptr, kernel_name) {\
    int __page_n = EXT_PAGE_N(ext_len);\
    for (int target = exec_id; target < __page_n; target += exec_total) {\
        int ld_value, w_cnt;\
        int* _check_ptr = reinterpret_cast<int*>(CHK_POSITION(recv_buf, ext_len, target, __page_n)) ;\
        while (true) {\
            ld_value = ld_acquire_sys_global(_check_ptr);\
            if (ld_value == ZTAG(tagv)) {\
                break;\
            }\
            if (count_value == 0) {\
                if (warp_id == 0) {\
                    count_value = ld_relaxed_sys_global(count_ptr);\
                    count_value = ((count_value & 0xffff0000) == SHORT_TAG(tagv)) ? (count_value | 0xffff0000) : 0;\
                    if (count_value != 0) st_release_cta(count_cache_ptr, count_value);\
                } else {\
                    count_value = ld_acquire_cta(count_cache_ptr);\
                }\
                if (count_value != 0 && token_idx >= (-count_value-1)) {\
                    break;\
                }\
            }\
            w_cnt += 1;\
            if (w_cnt == FINAL_TIME_MASK) {\
                printf("[rank %d]: [EAGER " kernel_name " HANG] round 0x%08x expert %3d from rank %d slot %d wait tag at offset %lu for %d times, 0x%08x != 0x%08x, cnt = %d(%d)\n", rank, tagv, rank * num_local_experts + local_expert_idx, src_rank, i, PTR_DIFF(_check_ptr, recv_buf), w_cnt, ld_value, ZTAG(tagv), count_value, -count_value-1);\
                break;\
            }\
        };\
    }\
    __syncwarp();\
}

#define NORMAL_ST(PTR, VALUE) *(PTR) = VALUE
#define NORMAL_LD(PTR) *(PTR)

#define LD_SHIFTED(LD_FUNC, SRC_PTR, SRC_BASE) LD_FUNC(SHIFTED_ADDR_P(SRC_PTR, SRC_BASE))

#define ST_SHIFTED(ST_FUNC, DST_PTR, DST_BASE, VALUE) ST_FUNC(SHIFTED_ADDR_P(DST_PTR, DST_BASE), VALUE)

#define N_LD_SHIFTED(SRC_PTR, SRC_BASE) LD_SHIFTED(NORMAL_LD, SRC_PTR, SRC_BASE)

#define N_ST_SHIFTED(DST_PTR, VALUE, DST_BASE) ST_SHIFTED(NORMAL_ST, DST_PTR, DST_BASE, VALUE)

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

#define UNROLLED_WARP_COPY_SRC_AUTO_SHIFT(UNROLL_FACTOR, LANE_ID, N, DST, SRC, SRC_RDMA_HEAD, LD_FUNC, ST_FUNC) \
{ \
    constexpr int kLoopStride = 32 * (UNROLL_FACTOR); \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)]; \
    auto __src = (SRC); \
    auto __dst = (DST); \
    for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) {\
            unrolled_values[__j] = LD_SHIFTED(LD_FUNC, __src + __i + __j * 32, SRC_RDMA_HEAD); \
        }\
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]); \
    } \
    for (int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); __i < (N); __i += 32) {\
        unrolled_values[0] = LD_SHIFTED(LD_FUNC, __src + __i, SRC_RDMA_HEAD);\
        ST_FUNC(__dst + __i, unrolled_values[0]); \
    }\
}

#define TMA_AUTO_TAG(tma_func, smem_ptr, gmem_ptr, bytes, gmem_base, tag_save, tagv, intra_node) {\
    if (!intra_node) {\
        const auto diff = PTR_DIFF(gmem_ptr, gmem_base);\
        int __BASE_PN__ = diff >> PCIE_SEG_LEN_LOG;\
        int __TAIL_PN__ = (diff + bytes) >> PCIE_SEG_LEN_LOG;\
        if (lane_id == 0 && __BASE_PN__ != __TAIL_PN__) {\
            int tag_tma_offset = (PCIE_SEG_LEN - PCIE_TAIL_SZ) - (diff & PCIE_SEG_LEN_MASK);\
            tag_save[__BASE_PN__] = *reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(smem_ptr) + tag_tma_offset);\
            st_release_cta(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(smem_ptr) + tag_tma_offset), ZTAG(tagv));\
        }\
    }\
    if (lane_id == 0) tma_func(smem_ptr, gmem_ptr, bytes);\
}

#define TMA_SAVE_TAG(kparam, intra_node, cpy_dst_int4_ptr, hidden_int4, long_msg_ext_len_int4, tail_tags, combine_round_n, num_send_bytes) {\
    if constexpr (kparam != EAGER_OFF) {\
        if (!intra_node) {\
            __syncwarp();\
            _Pragma("unroll") \
            for (int __pn = 0; __pn < MAX_PAGES; ++__pn) {\
                reinterpret_cast<int*>(tail_tags)[__pn] = __shfl_sync(0xffffffff, reinterpret_cast<int*>(tail_tags)[__pn], 0);\
            }\
            if (lane_id < MAX_PAGES_DIV4) {\
                auto target_ptr = cpy_dst_int4_ptr + hidden_int4 + lane_id;\
                *target_ptr = reinterpret_cast<int4*>(tail_tags)[lane_id];\
            }\
            if (lane_id == 0) {\
                *reinterpret_cast<int*>(cpy_dst_int4_ptr + long_msg_ext_len_int4 - 1) = ZTAG(combine_round_n);\
            }\
            __syncwarp();\
            num_send_bytes = long_msg_ext_len_int4 * sizeof(int4);\
        }\
    }\
}

#define TMA_RESTORE_TAG(kparam, logfmt_param, intra_node, smem_token_ptr, ext_len, decode_warp_idx, group_idx, num_decode_warps) {\
    if constexpr (kparam != EAGER_OFF) {\
        constexpr int pages = EXT_PAGE_N(ext_len);\
        if (!intra_node) {\
            if constexpr (logfmt_param) {\
                if (decode_warp_idx == 0) {\
                    int data_len = lane_id == 0 ? (*reinterpret_cast<int*>(smem_token_ptr) & 0xffff) : 0;\
                    data_len = __shfl_sync(0xffffffff, data_len, 0);\
                    const int pages = EXT_PAGE_N(data_len + (MAX_PAGES_DIV4 + 1) * sizeof(int4));\
                    if (lane_id < pages - 1) {\
                        const auto ld_offset = lane_id < 2 ? (sizeof(int) * (lane_id + 2)) : (data_len + (MAX_PAGES + lane_id - 1) * sizeof(int));\
                        const auto st_offset = (lane_id << PCIE_SEG_LEN_LOG) + (PCIE_SEG_LEN - PCIE_TAIL_SZ);\
                        int save_value = *(reinterpret_cast<int*>(smem_token_ptr + ld_offset));\
                        st_release_cta(reinterpret_cast<int*>(smem_token_ptr + st_offset), save_value);\
                    }\
                }\
            } else {\
                if (decode_warp_idx == 0 && lane_id < pages - 1) {\
                    const auto ld_offset = kHidden * sizeof(nv_bfloat16) + sizeof(int) * lane_id;\
                    const auto st_offset = ((lane_id << PCIE_SEG_LEN_LOG) + (PCIE_SEG_LEN - PCIE_TAIL_SZ));\
                    int save_value = *(reinterpret_cast<int*>(smem_token_ptr + ld_offset));\
                    st_release_cta(reinterpret_cast<int*>(smem_token_ptr + st_offset), save_value);\
                }\
            }\
            asm volatile("bar.sync %0, %1;" :: "r"(group_idx + 2), "r"(num_decode_warps * 32));\
        }\
    }\
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