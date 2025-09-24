
#ifndef __EAGER_SUPPORT_CU_H__
#define __EAGER_SUPPORT_CU_H__

#include "eager.h"

namespace deep_ep {

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

__device__  __forceinline__ void st_release_cta(const uint64_t *ptr, uint64_t val) {
    asm volatile("st.release.cta.u64 [%0], %1;"::"l"(ptr), "l"(val) : "memory");
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
#define PTR_DIFF(a, b) (reinterpret_cast<const uint8_t*>(a) - reinterpret_cast<const uint8_t*>(b))
#define SHIFTED_ADDR_P(ptr, bptr) ((ptr) + DIV4080(PTR_DIFF(ptr, bptr)) * PCIE_TAIL_SZ / sizeof(decltype(*(ptr))))
#define IS_PAGE_SUB_HEAD(ptr, bptr, size) (((PTR_DIFF(ptr, bptr) == (size)) || (PTR_DIFF(ptr, bptr) % (PCIE_SEG_LEN - PCIE_TAIL_SZ)) == 0))
#define CHK_POSITION(bptr, ext_size, pn, ptotal) (reinterpret_cast<uint8_t*>(bptr) + ((pn == ptotal - 1) ? (ext_size - PCIE_TAIL_SZ) : (((pn) << PCIE_SEG_LEN_LOG) + (PCIE_SEG_LEN - PCIE_TAIL_SZ))))
#define EXT_PAGE_N(size) ((size >> PCIE_SEG_LEN_LOG) + ((size & PCIE_SEG_LEN_MASK) != 0))

#define TAG_V_OFFSET 0
#define ZTAG(tag) (tag + TAG_V_OFFSET)
#define TAG_CNT_MASK 0x3fffffff
#define TAG_TYPE(tag) ((tag >> 31) & 1)
#define SHORT_TAG(tag) (((TAG_TYPE(tag) << 15) | ((((tag) & TAG_CNT_MASK) % 0x7fff) + 1)) << 16)
#define CHECK_TIME_MASK 0xffffff
#define FINAL_TIME_MASK 0x1000000


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
                printf("[rank %d]: [EAGER " kernel_name " HANG] round 0x%08x token %d topk %d from/to exp %d at rank %d, check offset %lu/%lu, %d times, 0x%08x != 0x%08x\n", rank, tagv, token_idx, topk_i, topk_idx, exp_rank, PTR_DIFF(__check_ptr, recv_buf), (uint64_t)ext_len, w_cnt, ld_value, ZTAG(tagv));\
                trap();\
            }\
        }\
    }\
    __syncwarp();\
}

#define WARP_WAIT_LEN_AND_BIT(recv_buf, head_pack, tagv, token_idx, topk_i, topk_idx, exp_rank, kernel_name) {\
    uint64_t head_value, data_len;\
    if (lane_id == 0) {\
        int w_cnt = 0;\
        while ((int)(head_value = ld_acquire_sys_global(reinterpret_cast<uint64_t*>(recv_buf)))!= ZTAG(tagv)) {\
            w_cnt += 1;\
            if (w_cnt == FINAL_TIME_MASK) {\
                printf("[rank %d]: [EAGER " kernel_name " HANG] round 0x%08x token %d topk %d recv from expert %d at rank %d, can not get header for logfmt message, 0x%08x != 0x%08x\n", rank, tagv, token_idx, topk_i, topk_idx, exp_rank, (int)(head_value & 0xffffffff), ZTAG(tagv));\
                trap();\
            }\
        }\
        data_len = head_value >> 32;\
        /*printf("[rank %d]: combine round 0x%08x token %d topk %d recv from expert %d at rank %d, len: %lu\n", rank, tagv, token_idx, topk_i, topk_idx, exp_rank, data_len)*/;\
        st_release_cta(reinterpret_cast<uint64_t*>(head_pack), ld_nc_global(reinterpret_cast<uint64_t*>(recv_buf) + 1));\
        st_release_cta(head_pack + 2, (int)data_len);\
    }\
    data_len = __shfl_sync(0xffffffff, data_len, 0);\
    int ext_len = data_len + 2 * sizeof(int4);\
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
                trap();\
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
            int* tag_tma_ptr = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(smem_ptr) + tag_tma_offset);\
            tag_save[__BASE_PN__] = *tag_tma_ptr;\
            /*printf("[rank %d]: combine round 0x%08x put %d tag (smem offset %d) on rank %d token %d, backup 0x%08x\n", rank, combine_round_n, __BASE_PN__, tag_tma_offset, dst_rank, src_idx, tag_save[__BASE_PN__])*/;\
            *tag_tma_ptr = ZTAG(tagv);\
            tma_store_fence();\
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

#define TMA_RESTORE_TAG(kparam, logfmt_param, intra_node, head_pack, data_offset, smem_token_ptr, ext_len, decode_warp_idx, group_idx, num_decode_warps) {\
    if constexpr (kparam != EAGER_OFF) {\
        constexpr int pages = EXT_PAGE_N(ext_len);\
        if (!intra_node) {\
            if constexpr (logfmt_param) {\
                if (decode_warp_idx == 0) {\
                    int data_len = lane_id == 0 ? head_pack[2] : 0;\
                    data_len = __shfl_sync(0xffffffff, data_len, 0);\
                    const int pages = EXT_PAGE_N(data_len + (MAX_PAGES_DIV4 + 1) * sizeof(int4));\
                    if (lane_id < pages - 1) {\
                        uint8_t* ld_ptr = lane_id < 2 ? reinterpret_cast<uint8_t*>(head_pack + lane_id) : (smem_token_ptr + data_len - data_offset + (lane_id - 1) * sizeof(int));\
                        uint8_t* st_ptr = smem_token_ptr + (lane_id << PCIE_SEG_LEN_LOG) + (PCIE_SEG_LEN - PCIE_TAIL_SZ) - data_offset - sizeof(int4);\
                        int value = *reinterpret_cast<int*>(ld_ptr);\
                        /*printf("[rank %d]: round 0x%08x token %d topk %d from expert %d at rank %d, restore two element at %lu with 0x%08x\n", rank, combine_round_n, token_idx, i, topk_idx, topk_idx / num_local_experts, PTR_DIFF(st_ptr, smem_token_ptr) / sizeof(nv_bfloat16), value)*/;\
                        st_release_cta(reinterpret_cast<int*>(st_ptr), value);\
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

#define EXT_SIZE_TIGHT(x) (x + (((x) >> PCIE_SEG_LEN_LOG) + 1) * sizeof(int))
#define TOKEN_OUT_OF_RANGE(idx, count) ((count) != 0 && ((idx) >= (-(count)-1)))


template <typename T>
__device__ __forceinline__ void Normal_ST(T *ptr, T& value) {
    *ptr = value;
}

template <typename T>
__device__ __forceinline__ T Normal_LD(const T *ptr) {
    return *ptr;
}

__device__ __forceinline__ void Default_Eager_Timeout_Func(const int* ptr, int value) {
    printf("[EAGER TAG CHECK TIMEOUT] ptr: %p, value: 0x%08x\n", ptr, value);
}

class EagerRDMASendBuffer {
    void *buf;
    size_t original_len;
    int tag_value;
public:
    void (*tag_st_func)(int *, int);
    __device__ EagerRDMASendBuffer(void *send_buf, size_t original_len, int tag_value, void (*int_st_func)(int*, int)): buf(send_buf), original_len(original_len), tag_value(tag_value), tag_st_func(int_st_func) {}
    __device__ EagerRDMASendBuffer(void *send_buf, size_t original_len, int tag_value): buf(send_buf), original_len(original_len), tag_value(tag_value), tag_st_func(nullptr) {}
    template <int kEager, typename T, typename Func, bool kNormalLDST = false>
    __device__ __forceinline__ void store(Func&& func, T* ptr, T value) {
        if constexpr (kEager != EAGER_OFF) {
            EP_STATIC_ASSERT(sizeof(T) >= sizeof(int), "can not support <4 byte element read/write");
            auto ptr_diff = PTR_DIFF(ptr, buf);
            if ((ptr_diff & PCIE_SEG_LEN_MASK) == (PCIE_SEG_LEN - PCIE_TAIL_SZ)) {
                auto st_ptr = &reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(buf) + original_len)[ptr_diff >> PCIE_SEG_LEN_LOG];
                auto *bp = reinterpret_cast<int*>(&value);
                if constexpr (kNormalLDST) {
                    *st_ptr = *bp;
                } else {
                    tag_st_func(st_ptr, *bp);
                }
                *bp = tag_value;
            }
            if (ptr_diff + sizeof(T) == original_len) {
                auto st_ptr = &reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(buf) + original_len)[original_len >> PCIE_SEG_LEN_LOG];
                if constexpr (kNormalLDST) {
                    *st_ptr = tag_value;
                } else {
                    tag_st_func(st_ptr, tag_value);
                }
            }
        }
        if constexpr (kNormalLDST) {
            *ptr = value;
        } else {
            func(ptr, value);
        }
    }
    template <int kEager, typename T>
    __device__ __forceinline__ void store(T* ptr, T& value) {
        this->store<kEager, T, decltype(&Normal_ST<T>), true>(&Normal_ST<T>, ptr, value);
    }
    template <int kEager, typename T, typename Func, bool kNormalLDST = false>
    __device__ __forceinline__ void shift_store(Func&& func, T* ptr, T value) {
        if constexpr (kEager != EAGER_OFF) {
            EP_STATIC_ASSERT(sizeof(T) >= sizeof(int), "can not support <4 byte element read/write");
            if constexpr (kNormalLDST) {
                N_ST_SHIFTED(ptr, value, buf);
            } else {
                ST_SHIFTED(func, ptr, buf, value);
            }
        } else {
            if constexpr (kNormalLDST) {
                *ptr = value;
            } else {
                func(ptr, value);
            }
        }
    }
    template <int kEager, typename T>
    __device__ __forceinline__ void shift_store(T* ptr, T& value) {
        this->shift_store<kEager, T, decltype(&Normal_ST<T>), true>(&Normal_ST<T>, ptr, value);
    }
    template <int kEager, bool kNormalLDST = false>
    __device__ __forceinline__ void shift_tag(int exec_id, int exec_total) {
        if constexpr (kEager != EAGER_OFF) {
        auto ext_len = EXTEND_FOR_TAG_AND_ALIGN(original_len, AR_MSG_ALIGNMENT);
            if (!kNormalLDST) {
                PARALLEL_SET_TAG(buf, ext_len, tag_value, exec_id, exec_total, tag_st_func);
            } else {
                PARALLEL_SET_TAG(buf, ext_len, tag_value, exec_id, exec_total, NORMAL_ST);
            }
        }
    }
    
    __device__ __forceinline__ void* buffer() {
        return buf;
    }

    __device__ __forceinline__ size_t len() {
        return original_len;
    }

    __device__ __forceinline__ int tag_v() {
        return tag_value;
    }
};

class EagerRDMARecvBuffer {
    void *buf;
    size_t original_len;
    int tag_value;
    int (*int_load_func)(const int*);
public:
    __device__ EagerRDMARecvBuffer(void *recv_buf, size_t original_len, int tag_value, int (*int_load_func)(const int*) = nullptr): buf(recv_buf), original_len(original_len), tag_value(tag_value), int_load_func(int_load_func) {}
    template <int kEager, typename T, typename Func, bool kNormalLDST = false>
    __device__ __forceinline__ T load(T* ptr, Func&& func) {
        std::remove_const_t<T> ld_value = func(ptr);
        if constexpr (kEager != EAGER_OFF) {
            auto ptr_diff = PTR_DIFF(ptr, buf);
            if ((ptr_diff & PCIE_SEG_LEN_MASK) >= (PCIE_SEG_LEN - PCIE_TAIL_SZ) && (ptr_diff & PCIE_SEG_LEN_MASK) < (PCIE_SEG_LEN - PCIE_TAIL_SZ + sizeof(T))) {
                auto ld_ptr = &reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(buf) + original_len)[ptr_diff >> PCIE_SEG_LEN_LOG];
                int got;
                if constexpr (kNormalLDST) {
                    got = *ld_ptr;
                } else {
                    got = int_load_func(ld_ptr);
                }
                if constexpr (sizeof(T) >= sizeof(int)) {
                    *reinterpret_cast<int*>(&ld_value) = got;
                } else {
                    ld_value = static_cast<T>(got >> ((ptr_diff & PCIE_SEG_LEN_MASK) - (PCIE_SEG_LEN - PCIE_TAIL_SZ)));
                }
            }
        }
        return ld_value;
    }
    template <int kEager, typename T>
    __device__ __forceinline__ T load(T *ptr) {
        return this->load<kEager, T, decltype(&Normal_LD<T>), true>(&Normal_LD<T>, ptr);
    }
    template <int kEager, typename T, typename Func, bool kNormalLDST = false>
    __device__ __forceinline__ T shift_load(T* ptr, Func&& func) {
        if constexpr (kEager != EAGER_OFF) {
            if constexpr (kNormalLDST) {
                return N_LD_SHIFTED(ptr, buf);
            } else {
                return LD_SHIFTED(func, ptr, buf);
            }
        } else {
            if constexpr (kNormalLDST) {
                return *ptr;
            } else {
                return func(ptr);
            }
        }
    }
    template <int kEager, typename T>
    __device__ __forceinline__ T shift_load(T *ptr) {
        return this->shift_load<kEager, T, decltype(&Normal_LD<T>), true>(&Normal_LD<T>, ptr);
    }
    template <int kEager, bool kCountCheck = false, typename PrintFunc>
    __device__ __forceinline__ void wait(int exec_id, int exec_total, PrintFunc func, int &count, int token_idx = 0, int *count_ptr = nullptr, int *count_cache_ptr = nullptr, bool count_use_cache = false) {
        if constexpr (kEager != EAGER_OFF) {
            int __page_n = (original_len >> PCIE_SEG_LEN_LOG) + 1;
            size_t ext_len = EXTEND_FOR_TAG_AND_ALIGN(original_len, AR_MSG_ALIGNMENT);
            for (int target = exec_id; target < __page_n; target += exec_total) {
                int ld_value, w_cnt = 0;
                int* __check_ptr = reinterpret_cast<int*>(CHK_POSITION(buf, ext_len, target, __page_n));
                while (true) {
                    ld_value = ld_acquire_sys_global(__check_ptr);
                    if (ld_value == ZTAG(tag_value)) break;
                    if constexpr (kCountCheck) {
                        if (count == 0) {
                            if (!count_use_cache) {
                                count = ld_relaxed_sys_global(count_ptr);
                                count = ((count & 0xffff0000) == SHORT_TAG(tag_value)) ? (count | 0xffff0000) : 0;
                                if (count != 0) {
                                    st_release_cta(count_cache_ptr, count);
                                }
                            } else {
                                count = ld_acquire_cta(count_cache_ptr);
                            }
                            if (TOKEN_OUT_OF_RANGE(token_idx, count)) {
                                break;
                            }
                        }
                    }
                    w_cnt += 1;
                    if (w_cnt == FINAL_TIME_MASK) {
                        func(__check_ptr, ld_value);
                        trap();
                    }
                }
            }
        }
    }

    __device__ __forceinline__ void* buffer() {
        return buf;
    }
};

#define EagerAutoAMO(ptr, value, dst_pe, qp_id) {\
    if constexpr (kEager <= EAGER_OFF) {\
        nvshmemi_ibgda_amo_nonfetch_add(ptr, value, dst_pe, qp_id);\
    } else {\
        nvshmemi_ibgda_rma_p(ptr, value, dst_pe, qp_id);\
    }\
}

#define E_WRAPPER_SHIFT_LOAD(kEager, wrapper, func, ptr) (kEager != EAGER_OFF ? LD_SHIFTED(func, ptr, wrapper.buffer()) : func(ptr))
#define E_WRAPPER_SHIFT_STORE(kEager, wrapper, func, ptr, value) {\
    if constexpr (kEager != EAGER_OFF) {\
        ST_SHIFTED(func, ptr, wrapper.buffer(), value);\
    } else {\
        func(ptr, value);\
    }\
}
#define E_WRAPPER_SHIFT_TAG(wrapper, exec_id, exec_total) if constexpr (kEager != EAGER_OFF) PARALLEL_SET_TAG(wrapper.buffer(), EXTEND_FOR_TAG_AND_ALIGN(wrapper.len(), AR_MSG_ALIGNMENT), wrapper.tag_v(), exec_id, exec_total, wrapper.tag_st_func)

};

#endif // __AR_SUPPORT_H__