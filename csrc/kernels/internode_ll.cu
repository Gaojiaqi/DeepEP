#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "ibgda_device.cuh"

#include "eager.cuh"
#include "utils.cuh"
#include <cstdint>
#include <cstdio>

namespace deep_ep {

namespace internode_ll {

template <int kNumThreads> __launch_bounds__(kNumThreads, 1)
__global__ void clean_low_latency_buffer(int* clean_0, int num_clean_int_0,
                                         int* clean_1, int num_clean_int_1) {
    // Barrier before cleaning (in case of unfinished chunked EP)
    nvshmemx_barrier_all_block();

    // Clean
    auto thread_id = static_cast<int>(threadIdx.x);
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_0; i += kNumThreads)
        clean_0[i] = 0;
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_1; i += kNumThreads)
        clean_1[i] = 0;

    // Barrier after cleaning (make sure the low-latency mode works fine)
    nvshmemx_barrier_all_block();
}

void clean_low_latency_buffer(int* clean_0, int num_clean_int_0,
                              int* clean_1, int num_clean_int_1,
                              cudaStream_t stream) {
    constexpr int kNumThreads = 256;

    SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, clean_low_latency_buffer<kNumThreads>,
                  clean_0, num_clean_int_0, clean_1, num_clean_int_1);
}


template <bool kUseFP8, bool kUseUE8M0, int kHidden>
__device__ __inline__ bool eager_dispatch_recv_token(
    void *rdma_recv_x, 
    int* rdma_recv_count,
    int responsible_expert_idx,
    int num_local_experts, 
    int num_experts,
    int rank, 
    int num_ranks, 
    int num_max_dispatch_tokens_per_rank, 
    size_t msg_distance,
    size_t num_bytes_per_msg,
    size_t num_bytes_per_msg_v,
    void* packed_recv_x, 
    void* packed_recv_x_scales,
    int* packed_recv_src_info,
    int64_t* packed_recv_layout_range,
    int* packed_recv_count,
    int* per_rank_recv_count,
    int& num_recv_tokens,
    int i,
    int dispatch_round_n
) {
    using scale_t = std::conditional_t<kUseUE8M0, uint8_t, float>;
    using packed_t = std::conditional_t<kUseUE8M0, uint32_t, float>;
    const auto lane_id = get_lane_id();
    constexpr int kNumPerChannels = 128;
    constexpr int num_scales = kHidden / kNumPerChannels;
    const size_t hidden_bytes = kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
    const size_t hidden_int4 = hidden_bytes / sizeof(int4);
    const auto src_rank = responsible_expert_idx / num_local_experts;
    const auto local_expert_idx = responsible_expert_idx % num_local_experts;
    const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
    const auto rdma_recv_x_uint8 = static_cast<uint8_t*>(rdma_recv_x) +
            local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * msg_distance +
            src_rank * num_max_dispatch_tokens_per_rank * msg_distance;
    const auto recv_x_int4 = static_cast<int4*>(packed_recv_x) +
            local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_int4;
    const auto recv_src_info = packed_recv_src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
    //const auto recv_range = packed_recv_layout_range + local_expert_idx * num_ranks;
    const auto token_src_bitmap = reinterpret_cast<int*>(packed_recv_layout_range) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank + src_rank * num_max_dispatch_tokens_per_rank; // used in EAGER_FULL
    const auto per_rank_recv_cnt_ptr = per_rank_recv_count + local_expert_idx * num_ranks; // used in EAGER_FULL
    const auto ld_intra_node = (src_rank >> 3) == (rank >> 3); // TODO: identify intra node more elegantly
    const auto num_aligned_scales = align<int>(num_scales, sizeof(float) / sizeof(scale_t));
    const auto recv_x_scales = static_cast<scale_t*>(packed_recv_x_scales) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_aligned_scales;

    const auto src_src_idx = reinterpret_cast<int*>(rdma_recv_x_uint8 + i * msg_distance);
    //const auto src_topk_idx = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));

    auto rcv_cnt_ptr = rdma_recv_count + local_expert_idx * num_ranks + src_rank;
    int this_token_index_in_recv = 0, shfl_value;
    int ready = 1;
    constexpr int kEager = EAGER_FULL;
    
    TRY_2BIT(src_src_idx, num_bytes_per_msg_v, num_bytes_per_msg, lane_id, 32, dispatch_round_n, rcv_cnt_ptr, num_recv_tokens, ready, i, ld_intra_node);
    ready = warp_reduce_min(ready);
    if (ready == 0) return false; // this token is not ready, go to next token
    
    if (lane_id == 0) {
        shfl_value = atomic_add_release_global(packed_recv_count + local_expert_idx, 1);
        int zzz = atomicAdd(per_rank_recv_cnt_ptr + src_rank, 1);
        //printf("[rank %d]: dispatch recv token expert %d slot %d get rank %d token %d, it is %d-th\n", rank, global_expert_idx, shfl_value, src_rank, LD_SHIFTED(ld_nc_global, src_src_idx, src_src_idx), zzz);
        token_src_bitmap[i] = shfl_value;
    }
    __syncwarp();

    this_token_index_in_recv = __shfl_sync(0xffffffff, shfl_value, 0);
    
#define DISPATCH_LD(ld_func, ptr) ((kEager != EAGER_OFF) ? LD_SHIFTED(ld_func, ptr, src_src_idx) : ld_func(ptr))

    if (lane_id == 0)
        recv_src_info[this_token_index_in_recv] = DISPATCH_LD(ld_nc_global, src_src_idx);
    __syncwarp();

    // Copy data
    // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
    const auto src_data = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
    const auto dst_data = recv_x_int4 + this_token_index_in_recv * hidden_int4;
    //if (kEager != EAGER_OFF) {
    UNROLLED_WARP_COPY_SRC_AUTO_SHIFT(5, lane_id, hidden_int4, dst_data, src_data, src_src_idx, ld_nc_global, st_na_global);  
    // } else {
    //     UNROLLED_WARP_COPY(7, lane_id, hidden_int4, dst_data, src_data, ld_nc_global, st_na_global);
    // }

    // Copy scales
    if constexpr (kUseFP8) {
        // Equivalent CuTe layout:
        //   (num_tokens, (num_packed, num_elems_per_pack)):(num_elems_per_pack, (num_tokens * num_elems_per_pack, 1))
        const auto src_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
        const auto num_elems_per_pack = static_cast<int>(sizeof(packed_t) / sizeof(scale_t));
        const auto token_idx = this_token_index_in_recv;
        const auto token_stride = num_elems_per_pack;
        const auto pack_stride = num_ranks * num_max_dispatch_tokens_per_rank * num_elems_per_pack;
        if (lane_id < num_scales) {
            const auto pack_idx = lane_id / num_elems_per_pack;
            const auto elem_idx = lane_id % num_elems_per_pack;
            auto scale = extract_required_scale_format<kUseUE8M0>(DISPATCH_LD(ld_nc_global, src_scales + lane_id));
            recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
        }
        if (lane_id + 32 < num_scales) {
            const auto pack_idx = (lane_id + 32) / num_elems_per_pack;
            const auto elem_idx = (lane_id + 32) % num_elems_per_pack;
            auto scale = extract_required_scale_format<kUseUE8M0>(DISPATCH_LD(ld_nc_global, src_scales + lane_id + 32));
            recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
        }
    }
    return true; // got a token
}

template <bool kUseFP8, bool kUseUE8M0, int kHidden, int kEager>
__global__ __launch_bounds__(1024, 1) void
dispatch(void* packed_recv_x, void* packed_recv_x_scales,
         int* packed_recv_src_info, int64_t* packed_recv_layout_range,
         int* packed_recv_count, int* per_rank_recv_count,
         int* cumulative_local_expert_recv_stats,
         int64_t* dispatch_wait_recv_cost_stats,
         void* rdma_recv_x, int* rdma_recv_count, void* rdma_x,
         const void* x, const int64_t* topk_idx,
         int* atomic_counter_per_expert, int* atomic_finish_counter_per_expert,
         int* next_clean, int num_next_clean_int,
         int num_tokens, int num_max_dispatch_tokens_per_rank,
         int num_topk, int num_experts, int rank, int num_ranks,
         int num_warp_groups, int num_warps_per_group,
         bool round_scale, int phases,
         int dispatch_round_n) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

    // May extract UE8M0 from the scales
    using scale_t = std::conditional_t<kUseUE8M0, uint8_t, float>;
    using packed_t = std::conditional_t<kUseUE8M0, uint32_t, float>;
    EP_STATIC_ASSERT(sizeof(packed_t) % sizeof(scale_t) == 0, "Invalid vector length");

    // FP8 staffs
    constexpr int kNumPerChannels = 128;
    constexpr int num_scales = kHidden / kNumPerChannels;
    const size_t hidden_bytes = kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
    const size_t hidden_int4 = hidden_bytes / sizeof(int4);

    // Message package: hidden data, FP8 scales, index at source
    // NOTES: currently we have 3 reserved int fields for future use
    using vec_t = std::conditional_t<kUseFP8, int2, int4>;
    const size_t num_bytes_per_msg_v = sizeof(int4) + (kUseFP8 ? (kHidden + num_scales * sizeof(float)) : (kHidden * sizeof(nv_bfloat16)));
    const size_t num_bytes_per_msg = kEager != EAGER_OFF ? EXTEND_FOR_TAG_AND_ALIGN(num_bytes_per_msg_v, AR_MSG_ALIGNMENT) : num_bytes_per_msg_v;
    
    // WARNING!!! Two lines below must be consistent with msg def in config.hpp
    constexpr size_t dispatch_msg_max = sizeof(int4) + std::max(kHidden * sizeof(nv_bfloat16), kHidden + num_scales * sizeof(float));
    constexpr size_t combine_msg_max = num_scales * sizeof(nv_bfloat162) + kHidden * sizeof(nv_bfloat16);
    constexpr size_t msg_distance = kEager != EAGER_OFF ? EXTEND_FOR_TAG_AND_ALIGN(std::max(dispatch_msg_max, combine_msg_max) + sizeof(int), AR_MSG_LONG_ALIGNMENT) : num_bytes_per_msg_v;

    const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
    EP_DEVICE_ASSERT(num_bytes_per_msg % sizeof(int4) == 0);
    
    //if (sm_id == 0 && thread_id == 0) printf("[rank %d]: kEager = %d\n", rank, kEager);
    // Expert counts
    constexpr int kNumMaxWarpGroups = 32;
    __shared__ int shared_num_tokens_sent_per_expert[kNumMaxWarpGroups];

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_DISPATCH_RECV;

    // There are 2 kinds of warps in this part:
    // 1. The first-kind warps for FP8 cast and sending top-k tokens
    // 2. The last warp for reading `topk_idx` and count for per-expert information
    if (warp_id < num_warps - 1) {
        constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
        EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerRead) == 0, "Invalid hidden");
        EP_STATIC_ASSERT(kNumElemsPerRead * 32 % kNumPerChannels == 0, "Invalid vectorization");
        const auto num_threads = (num_warps - 1) * 32;
        const size_t hidden_bf16_int4 = kHidden / kNumElemsPerRead;

        for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
            const auto x_int4 = static_cast<const int4*>(x) + token_idx * hidden_bf16_int4;
            const auto rdma_x_src_idx = reinterpret_cast<int*>(static_cast<uint8_t*>(rdma_x) + token_idx * num_bytes_per_msg);
            const auto rdma_x_vec = reinterpret_cast<vec_t*>(reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
            const auto rdma_x_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(rdma_x_vec) + hidden_bytes);

            // Overlap top-k index read and source token index writes
            auto dst_expert_idx = warp_id < num_topk ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id)) : -1;

#define DISPATCH_ST(PTR, VALUE) {\
    if (kEager != EAGER_OFF) {\
        N_ST_SHIFTED(PTR, VALUE, rdma_x_src_idx);\
    } else {\
        NORMAL_ST(PTR, VALUE);\
    }\
}

            if (thread_id == 0) {
                DISPATCH_ST(rdma_x_src_idx, token_idx);
            }

            // FP8 cast
            EP_STATIC_ASSERT(hidden_bf16_int4 % 32 == 0, "Must use the full warp to reduce");
            #pragma unroll
            for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
                // Read
                auto int4_value = __ldg(x_int4 + i);

                if constexpr (kUseFP8) {
                    // Calculate local amax
                    auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
                    float fp32_values[kNumElemsPerRead];
                    float amax = kFP8Margin, scale, scale_inv;
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; ++ j) {
                        fp32_values[j] = static_cast<float>(bf16_values[j]);
                        amax = fmaxf(amax, fabsf(fp32_values[j]));
                    }

                    // Reduce amax and scale
                    EP_STATIC_ASSERT(kNumElemsPerRead * 32 / kNumPerChannels == 2, "Invalid vectorization");
                    amax = warp_reduce_max<16>(amax);
                    calculate_fp8_scales(amax, scale, scale_inv, round_scale);
                    if (lane_id == 0 or lane_id == 16) {
                        DISPATCH_ST(&rdma_x_scales[i * kNumElemsPerRead / 128], scale_inv);
                    }

                    // Cast into send buffer
                    vec_t int2_value;
                    auto fp8x2_values = reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; j += 2) {
                        float2 fp32x2 = {fp32_values[j] * scale, fp32_values[j + 1] * scale};
                        fp8x2_values[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
                    }
                    DISPATCH_ST(&rdma_x_vec[i], int2_value);
                } else {
                    // Reinterpret-cast is for C++14 compatibility
                    DISPATCH_ST(reinterpret_cast<int4*>(&rdma_x_vec[i]), int4_value);
                }
            }
            if (kEager != EAGER_OFF) {
                const auto warp_runs = hidden_bf16_int4 / 32;
                const auto idle_warp_id = (warp_id + (num_warps - 1) - (warp_runs % (num_warps - 1))) % (num_warps - 1);
                if (idle_warp_id == 0) {
                    PARALLEL_SET_TAG(rdma_x_src_idx, num_bytes_per_msg_v, dispatch_round_n, lane_id, 32, NORMAL_ST);
                }
            }
            asm volatile("bar.sync 1, %0;" :: "r"(num_threads));

            // Issue IBGDA sends
            if (dst_expert_idx >= 0) {
                int slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;
                slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);
                const auto dst_rank = dst_expert_idx / num_local_experts;
                const auto dst_expert_local_idx = dst_expert_idx % num_local_experts;
                const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                                     dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * msg_distance +
                                     rank * num_max_dispatch_tokens_per_rank * msg_distance +
                                     slot_idx * msg_distance;
                const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
                if (dst_p2p_ptr == 0) {
                    if constexpr (kEager == EAGER_FULL) {
                        slot_idx = -1;
                    }
                    nvshmemi_ibgda_put_nbi_warp(dst_ptr, src_ptr, num_bytes_per_msg, dst_rank, dst_expert_local_idx, lane_id, slot_idx);
                } else {
                    // NOTES: only 2 load iterations for 7K hidden with 8 unrolls
                    const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
                    const auto* dst_int4_ptr = reinterpret_cast<int4*>(dst_p2p_ptr);
                    UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg, dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
                    WARP_SET_TAIL_TAG(dst_p2p_ptr, num_bytes_per_msg, dispatch_round_n);
                }

                // Increase counter after finishing
                if (kEager != EAGER_FULL) {
                    __syncwarp();
                    lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1) : 0;
                }
            }
        }
    } else if (warp_id == num_warps - 1) {
        EP_DEVICE_ASSERT(num_sms > 1);
        if (sm_id == 0) {
            // The first SM is also responsible for checking QPs
            EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= num_local_experts);

            // The first SM is also responsible for cleaning the next buffer
            #pragma unroll
            for (int i = lane_id; i < num_next_clean_int; i += 32)
                next_clean[i] = 0;

            // Notify before executing `int_p`
            __syncwarp();
            if (kEager != EAGER_FULL) {
                #pragma unroll
                for (int i = lane_id; i < num_experts; i += 32)
                    atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG);
            }
        }

        // This SM should be responsible for some destination experts, read `topk_idx` for them
        int expert_count[kNumMaxWarpGroups] = {0};
        const auto expert_begin_idx = sm_id * num_warp_groups;
        const auto expert_end_idx = min(expert_begin_idx + num_warp_groups, num_experts);

        // Per lane count
        #pragma unroll 8
        for (int i = lane_id; i < num_tokens * num_topk; i += 32) {
            auto idx = static_cast<int>(__ldg(topk_idx + i));
            if (idx >= expert_begin_idx and idx < expert_end_idx)
                expert_count[idx - expert_begin_idx] ++;
        }

        // Warp reduce
        #pragma unroll
        for (int i = expert_begin_idx; i < expert_end_idx; ++ i) {
            auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
            if (lane_id == 0) {
                if (kEager == EAGER_FULL) {
                    const auto dst_rank = i / num_local_experts;
                    const auto dst_expert_local_idx = i % num_local_experts;
                    const auto num_tokens_sent = -sum - 1;
                    auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_count + dst_expert_local_idx * num_ranks + rank);
                    auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
                    //if (sum != 0) printf("[rank %d]: %d tokens to expert %d\n", rank, sum, i);
                    if (dst_p2p_ptr == 0) {
                        nvshmemi_ibgda_rma_p(reinterpret_cast<int*>(dst_ptr), num_tokens_sent, dst_rank, dst_expert_local_idx);
                    } else {
                        st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), num_tokens_sent);
                    }
                } else {
                    shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
                    atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG - sum);
                }
            }
        }
    }

    if (kEager != EAGER_FULL) {
        __syncthreads();

        // Issue count sends
        if (responsible_expert_idx < num_experts and sub_warp_id == 0 and lane_id == 0) {
            const auto dst_rank = responsible_expert_idx / num_local_experts;
            const auto dst_expert_local_idx = responsible_expert_idx % num_local_experts;
            const auto num_tokens_sent = shared_num_tokens_sent_per_expert[responsible_expert_idx - sm_id * num_warp_groups];

            // Wait local sends issued and send expert counts
            while (ld_acquire_global(atomic_finish_counter_per_expert + responsible_expert_idx) != FINISHED_SUM_TAG * 2);
            auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_count + dst_expert_local_idx * num_ranks + rank);
            auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
            if (dst_p2p_ptr == 0) {
                if (kEager <= EAGER_OFF) {
                    nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), -num_tokens_sent - 1, dst_rank, dst_expert_local_idx);
                } else {
                    nvshmemi_ibgda_rma_p(reinterpret_cast<int*>(dst_ptr), -num_tokens_sent - 1, dst_rank, dst_expert_local_idx);
                }
            } else {
                st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), -num_tokens_sent - 1);
            }

            // Clean workspace for next use
            atomic_counter_per_expert[responsible_expert_idx] = 0;
            atomic_finish_counter_per_expert[responsible_expert_idx] = 0;

            // Clean `packed_recv_count`
            if (dst_rank == 0)
                packed_recv_count[dst_expert_local_idx] = 0;
        }
        __syncwarp();
    }

    // Receiving phase
    LOW_LATENCY_DISPATCH_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // For send-and-recv kernels, we need a grid sync for making `packed_recv_count` visible
    if ((kEager != EAGER_FULL) && (phases & LOW_LATENCY_SEND_PHASE))
        cg::this_grid().sync();

    // Receiving and packing
    if (responsible_expert_idx < num_experts) {
        const auto src_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
        const auto rdma_recv_x_uint8 = static_cast<uint8_t*>(rdma_recv_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * msg_distance +
                src_rank * num_max_dispatch_tokens_per_rank * msg_distance;
        const auto recv_x_int4 = static_cast<int4*>(packed_recv_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_int4;
        const auto recv_src_info = packed_recv_src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto recv_range = packed_recv_layout_range + local_expert_idx * num_ranks;
        const auto token_src_bitmap = reinterpret_cast<int*>(packed_recv_layout_range) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank + src_rank * num_max_dispatch_tokens_per_rank; // used in EAGER_FULL
        const auto per_rank_recv_cnt_ptr = per_rank_recv_count + local_expert_idx * num_ranks; // used in EAGER_FULL
        const auto ld_intra_node = (src_rank >> 3) == (rank >> 3); // TODO: identify intra node more elegantly
        const auto num_aligned_scales = align<int>(num_scales, sizeof(float) / sizeof(scale_t));
        const auto recv_x_scales = static_cast<scale_t*>(packed_recv_x_scales) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_aligned_scales;

        // Shared between sub-warps in warp groups
        __shared__ int shared_num_recv_tokens[kNumMaxWarpGroups], shared_recv_token_begin_idx[kNumMaxWarpGroups];

        // Wait tokens to arrive
        // NOTES: using sub-warp 1 to overlap with sub-warp 0
        int num_recv_tokens = 0, recv_token_begin_idx;
        auto rcv_cnt_ptr = rdma_recv_count + local_expert_idx * num_ranks + src_rank;
        if constexpr (kEager != EAGER_FULL) {
            EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 15);
            if constexpr (kEager != EAGER_FULL) {
                if (sub_warp_id == 1 and lane_id == 0) {
                    auto start_time = clock64();
                    while ((num_recv_tokens = ld_acquire_sys_global(rdma_recv_count + local_expert_idx * num_ranks + src_rank)) == 0);
                    auto wait_recv_cost = clock64() - start_time;
                    num_recv_tokens = -num_recv_tokens - 1;
                    recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
                    shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
                    shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
                    recv_range[src_rank] = pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);

                    // Add stats for diagnosis
                    if (cumulative_local_expert_recv_stats != nullptr)
                        atomicAdd(cumulative_local_expert_recv_stats + local_expert_idx, num_recv_tokens);
                    if (dispatch_wait_recv_cost_stats != nullptr)
                        atomicAdd(reinterpret_cast<unsigned long long*>(dispatch_wait_recv_cost_stats + src_rank), wait_recv_cost);
                }
                asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 2), "r"(num_warps_per_group * 32));
                num_recv_tokens = shared_num_recv_tokens[warp_group_id];
                recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];
            }
            // Copy tokens
            EP_DEVICE_ASSERT(num_scales <= 64);
            int this_token_index_in_recv = 0;
            for (int i = sub_warp_id; (kEager != EAGER_FULL && i < num_recv_tokens) || (kEager == EAGER_FULL && (num_recv_tokens == 0 || i < -num_recv_tokens-1) && i < num_max_dispatch_tokens_per_rank); i += num_warps_per_group) {
                // Copy source info
                const auto src_src_idx = reinterpret_cast<int*>(rdma_recv_x_uint8 + i * msg_distance);

                if (kEager == EAGER_FULL) {
                    WAIT_2BIT(src_src_idx, num_bytes_per_msg_v, num_bytes_per_msg, lane_id, 32, dispatch_round_n, rcv_cnt_ptr, num_recv_tokens, i, ld_intra_node);
                    __syncwarp();
                    num_recv_tokens = warp_reduce_min(num_recv_tokens);
                    
                    if (num_recv_tokens != 0 && i >= (-num_recv_tokens-1)) {
                        //if (sub_warp_id == 0 && lane_id == 0 && (-num_recv_tokens-1) != 0) printf("[rank %d]: dispatch recv token expert %d get %d tokens from rank %d, done\n", rank, global_expert_idx, (-num_recv_tokens-1), src_rank);
                        break;
                    }
                    if (lane_id == 0) {
                        this_token_index_in_recv = atomic_add_release_global(packed_recv_count + local_expert_idx, 1);
                        atomicAdd(per_rank_recv_cnt_ptr + src_rank, 1);
                        //printf("[rank %d]: dispatch recv token expert %d slot %d get rank %d token %d\n", rank, global_expert_idx, this_token_index_in_recv, src_rank, LD_SHIFTED(ld_nc_global, src_src_idx, src_src_idx));
                        token_src_bitmap[i] = this_token_index_in_recv;
                    }
                    __syncwarp();
                    this_token_index_in_recv = __shfl_sync(0xffffffff, this_token_index_in_recv, 0);
                } else if constexpr (kEager != EAGER_OFF) {
                    WAIT_BIT(src_src_idx, num_bytes_per_msg_v, num_bytes_per_msg, lane_id, 32, dispatch_round_n, ld_intra_node);
                    __syncwarp();
                }
                if constexpr (kEager != EAGER_FULL) {
                    this_token_index_in_recv = recv_token_begin_idx + i;
                }

                if (lane_id == 0)
                    recv_src_info[this_token_index_in_recv] = DISPATCH_LD(ld_nc_global, src_src_idx);
                __syncwarp();

                // Copy data
                // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
                const auto src_data = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
                const auto dst_data = recv_x_int4 + this_token_index_in_recv * hidden_int4;
                if (kEager != EAGER_OFF) {
                    UNROLLED_WARP_COPY_SRC_AUTO_SHIFT(7, lane_id, hidden_int4, dst_data, src_data, src_src_idx, ld_nc_global, st_na_global);  
                } else {
                    UNROLLED_WARP_COPY(7, lane_id, hidden_int4, dst_data, src_data, ld_nc_global, st_na_global);
                }

                // Copy scales
                if constexpr (kUseFP8) {
                    // Equivalent CuTe layout:
                    //   (num_tokens, (num_packed, num_elems_per_pack)):(num_elems_per_pack, (num_tokens * num_elems_per_pack, 1))
                    const auto src_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
                    const auto num_elems_per_pack = static_cast<int>(sizeof(packed_t) / sizeof(scale_t));
                    const auto token_idx = this_token_index_in_recv;
                    const auto token_stride = num_elems_per_pack;
                    const auto pack_stride = num_ranks * num_max_dispatch_tokens_per_rank * num_elems_per_pack;
                    if (lane_id < num_scales) {
                        const auto pack_idx = lane_id / num_elems_per_pack;
                        const auto elem_idx = lane_id % num_elems_per_pack;
                        auto scale = extract_required_scale_format<kUseUE8M0>(DISPATCH_LD(ld_nc_global, src_scales + lane_id));
                        recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
                    }
                    if (lane_id + 32 < num_scales) {
                        const auto pack_idx = (lane_id + 32) / num_elems_per_pack;
                        const auto elem_idx = (lane_id + 32) % num_elems_per_pack;
                        auto scale = extract_required_scale_format<kUseUE8M0>(DISPATCH_LD(ld_nc_global, src_scales + lane_id + 32));
                        recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
                    }
                }
            }
            //if constexpr (kEager == EAGER_FULL) if ((-num_recv_tokens-1) != 0 && sub_warp_id < num_max_dispatch_tokens_per_rank && lane_id == 0) printf("[rank %d]: dispatch recv token expert %d get %d tokens from rank %d, subwarp %d done\n", rank, global_expert_idx, (-num_recv_tokens-1), src_rank, sub_warp_id);
        } else {
            // EAGER_FULL use soft warp scheduling to find ready tokens as soon as possible
            constexpr int hard_warps = 4; // hardware warp units
            const int working_queues = num_warps;
            __shared__ int current_token_index[32]; // working token index
            __shared__ int token_cnt_rcv[32]; // token count
            int still_working_queues = 0;
            int token_index = 0;
            if (warp_id < hard_warps) {
                for (int i = thread_id; i < working_queues; i += hard_warps * 32) {
                    current_token_index[i] = i % num_warps_per_group;
                    if (i < num_warp_groups) {
                        token_cnt_rcv[i] = 0;
                    }
                }
                asm volatile("bar.sync 1, %0;" :: "r"(hard_warps * 32));
                still_working_queues = (working_queues - warp_id - 1) / hard_warps + 1;
                while (still_working_queues > 0) {
                    for (int i = warp_id; i < working_queues; i += hard_warps) {
                        token_index = lane_id == 0 ? current_token_index[i] : 0; // pick a virtual warp
                        token_index = __shfl_sync(0xffffffff, token_index, 0);
                        auto rsp_exp_idx = (sm_id * num_warp_groups + (i / num_warps_per_group));
                        if (token_index < 0) { // virtual warp is already done
                            continue;
                        }
                        while (true) {
                            // try recv token
                            int num_recv_tokens = lane_id == 0 ? token_cnt_rcv[i / num_warps_per_group] : 0; // rank -> exp token cnt (-x-1 format)
                            num_recv_tokens = __shfl_sync(0xffffffff, num_recv_tokens, 0);
                            if (token_index >= num_max_dispatch_tokens_per_rank || (num_recv_tokens != 0 && token_index >= -num_recv_tokens-1)) { // out of range, so this virtual warp is done
                                token_index = -1; // this virtual warp is done
                                still_working_queues -= 1; // one less queue to work on
                                break;
                            }
                            int num_recv_tokens_ref = num_recv_tokens;
                            bool got_token = eager_dispatch_recv_token<kUseFP8, kUseUE8M0, kHidden>(rdma_recv_x, rdma_recv_count, rsp_exp_idx, num_local_experts, num_experts, rank, num_ranks, num_max_dispatch_tokens_per_rank, msg_distance, num_bytes_per_msg, num_bytes_per_msg_v, packed_recv_x, packed_recv_x_scales, packed_recv_src_info, packed_recv_layout_range, packed_recv_count, per_rank_recv_count, num_recv_tokens_ref, token_index, dispatch_round_n);
                            if (lane_id == 0 && num_recv_tokens == 0 && num_recv_tokens_ref != 0) {
                                // new count info
                                token_cnt_rcv[i / num_warps_per_group] = num_recv_tokens_ref;
                            }
                            __syncwarp();
                            if (!got_token) break; // no new token in this virtual warp, try next virtual warp
                            token_index += num_warps_per_group; // goto next token in this virtual warp
                        }
                        if (lane_id == 0) {
                            current_token_index[i] = token_index;
                        }
                        __syncwarp();
                    }
                }
            }
        }
    }
}

void dispatch(void* packed_recv_x, void* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range,
              int* packed_recv_count, int* per_rank_recv_count,
              int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats,
              void* rdma_recv_x, int* rdma_recv_count, void* rdma_x,
              const void* x, const int64_t* topk_idx,
              int* next_clean, int num_next_clean_int,
              int num_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
              int num_topk, int num_experts, int rank, int num_ranks,
              bool use_fp8, bool round_scale, bool use_ue8m0,
              void* workspace, int num_device_sms,
              cudaStream_t stream, int phases,
              int dispatch_round_n, const int eager_opt) {
    constexpr int kNumMaxTopK = 9;
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = 32 / num_warp_groups;
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);
    EP_HOST_ASSERT(kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms = ceil_div(num_experts, num_warp_groups);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopK);

    // Workspace checks
    auto atomic_counter_per_expert = static_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
    EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);
    if ((eager_opt == EAGER_FULL) && (phases & LOW_LATENCY_SEND_PHASE)) {
        CUDA_CHECK(cudaMemsetAsync(workspace, 0, num_experts * sizeof(int), stream)); // only set atomic_counter_per_expert
        CUDA_CHECK(cudaMemsetAsync(packed_recv_count, 0, sizeof(int) * num_experts / num_ranks, stream));
        CUDA_CHECK(cudaMemsetAsync(per_rank_recv_count, 0, sizeof(int) * num_experts, stream));
    }
    // FP8 checks
    if (use_ue8m0)
        EP_HOST_ASSERT(round_scale and "UE8M0 SF requires `round_scale=True`");

#define DISPATCH_LAUNCH_CASE(eager, hidden) { \
auto dispatch_func = dispatch<false, false, hidden, eager>; \
if (use_fp8 and not use_ue8m0) \
    dispatch_func = dispatch<true, false, hidden, eager>; \
if (use_fp8 and use_ue8m0) \
    dispatch_func = dispatch<true, true, hidden, eager>; \
LAUNCH_KERNEL(&cfg, dispatch_func, \
              packed_recv_x, packed_recv_x_scales, \
              packed_recv_src_info, packed_recv_layout_range, \
              packed_recv_count, per_rank_recv_count, \
              cumulative_local_expert_recv_stats, \
              dispatch_wait_recv_cost_stats, \
              rdma_recv_x, rdma_recv_count, rdma_x, \
              x, topk_idx, \
              atomic_counter_per_expert, atomic_finish_counter_per_expert, \
              next_clean, num_next_clean_int, \
              num_tokens, num_max_dispatch_tokens_per_rank, \
              num_topk, num_experts, rank, num_ranks, \
              num_warp_groups, num_warps_per_group, \
              round_scale, phases, dispatch_round_n); } break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);

    #define EAGER_DISPATCH_LAUNCH_CASE(hidden) SWITCH_EAGER(DISPATCH_LAUNCH_CASE, hidden)
    
    SWITCH_HIDDEN(EAGER_DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <int kNumSendUnrolls>
__forceinline__ __device__ int logfmt_encode(void* buffer, nv_bfloat162 *shared_amaxmin, const int& lane_id) {
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr float kLogThreshold = 0;
    constexpr float kMinClip = 32; // `== log_2(2 ^ (2 ^ 5))`
    constexpr int kNumBits = 10;
    constexpr int kNumValues = 1 << (kNumBits - 1);

    int4 int4_values[kNumSendUnrolls];
    const auto& uint32_values = reinterpret_cast<uint32_t*>(int4_values);
    const auto& bf162_values = reinterpret_cast<nv_bfloat162*>(int4_values);

    // Calculate lane offset
    const auto& ld_buffer = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(buffer) + lane_id * (kNumSendUnrolls * sizeof(int4)));
    const auto& st_buffer = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(buffer) + lane_id * (kNumSendUnrolls * sizeof(int4) * 10 / 16));

    // Local log amax
    auto bf162_amax = __nv_bfloat162(CUDART_ZERO_BF16, CUDART_ZERO_BF16);
    auto bf162_amin = __nv_bfloat162(CUDART_INF_BF16, CUDART_INF_BF16);
    uint32_t local_signs = 0;
    #pragma unroll
    for (int k = 0; k < kNumSendUnrolls * kNumElemsPerInt4 / 2; ++ k) {
        // TODO: eliminate bank conflicts
        uint32_values[k] = ld_buffer[k];
        local_signs |= ((uint32_values[k] >> 15) & 1) << (k * 2);
        local_signs |= ((uint32_values[k] >> 31) & 1) << (k * 2 + 1);
        uint32_values[k] &= 0x7fff7fff;

        bf162_amax = __hmax2(bf162_amax, bf162_values[k]);
        bf162_amin = __hmin2(bf162_amin, bf162_values[k]);
    }

    // Reduce per 128 channels
    // TODO: figure out how hardware do 2-byte min/max
    auto amax = std::max(static_cast<float>(bf162_amax.x), static_cast<float>(bf162_amax.y));
    auto amin = std::min(static_cast<float>(bf162_amin.x), static_cast<float>(bf162_amin.y));
    constexpr static int kNumLanesToReduce = 128 * sizeof(nv_bfloat16) / (kNumSendUnrolls * sizeof(int4));
    amax = warp_reduce_max<kNumLanesToReduce>(amax);
    amin = warp_reduce_min<kNumLanesToReduce>(amin);

    // Write min/max into the shared memory
    if (shared_amaxmin != nullptr)
        *shared_amaxmin = __nv_bfloat162(amax, amin);
    __syncwarp();

    // Calculate log amin/amax float
    const auto& log_amax = log2f_approx(amax);
    const auto& log_amin = fmaxf(log2f_approx(amin), log_amax - kMinClip);
    const bool& enable_cast = warp_reduce_and<kNumLanesToReduce, true>(log_amax < kLogThreshold and log_amin < log_amax);

    // Case into LogFMT-10 if satisfied
    if (enable_cast) {
        const auto step = (log_amax - log_amin) / static_cast<float>(kNumValues - 2);
        const auto step_inv = 1.0f / step;
        const auto rounding = 2.0f - log2f_approx((1.0f + exp2f_approx(step)) * 0.5f) * step_inv;
        const auto fused_rounding = rounding - log_amin * step_inv;

        // Pack every 256 bits into 160 bits
        EP_STATIC_ASSERT(kNumSendUnrolls == 2 or kNumSendUnrolls == 4, "kNumSendUnrolls == 2 or 4 only");
        uint32_t encoded[kNumElemsPerInt4 * 2];
        #pragma unroll 1
        for (int i = 0; i < kNumSendUnrolls / 2; ++ i) {
            #pragma unroll
            for (int k = 0; k < kNumElemsPerInt4; ++ k) {
                const auto& [x, y] = __bfloat1622float2(bf162_values[i * kNumElemsPerInt4 + k]);
                encoded[k * 2 + 0] = __float2uint_rd(fmaxf(log2f_approx(x) * step_inv + fused_rounding, 0));
                encoded[k * 2 + 1] = __float2uint_rd(fmaxf(log2f_approx(y) * step_inv + fused_rounding, 0));
            }
            st_buffer[i * 5 + 0] = (encoded[ 0] >> 0) | (encoded[ 1] << 9) | (encoded[ 2] << 18) | (encoded[ 3] << 27);
            st_buffer[i * 5 + 1] = (encoded[ 3] >> 5) | (encoded[ 4] << 4) | (encoded[ 5] << 13) | (encoded[ 6] << 22) | (encoded[7]  << 31);
            st_buffer[i * 5 + 2] = (encoded[ 7] >> 1) | (encoded[ 8] << 8) | (encoded[ 9] << 17) | (encoded[10] << 26);
            st_buffer[i * 5 + 3] = (encoded[10] >> 6) | (encoded[11] << 3) | (encoded[12] << 12) | (encoded[13] << 21) | (encoded[14] << 30);
            st_buffer[i * 5 + 4] = (encoded[14] >> 2) | (encoded[15] << 7) | ((i == 0) ? (local_signs << 16) : (local_signs & 0xffff0000u));
        }
        tma_store_fence();
        __syncwarp();
    }

    // Return TMA copy bytes
    return enable_cast ? (32 * (kNumSendUnrolls * sizeof(int4) * 8 * 10 / 16 / 8)):
                         (32 * (kNumSendUnrolls * sizeof(int4)));
}

template <int kNumLanes, int kNumSendUnrolls, int kNumRecvUnrolls>
__forceinline__ __device__ void logfmt_check_amaxmin(uint8_t* meta_buffer, float2* shared_log_amax,
                                                     float2* shared_log_amin, int* shared_cast_info,
                                                     const int lane_id) {
    constexpr float kLogThreshold = 0;
    constexpr float kMinClip = 32; // `== log_2(2 ^ (2 ^ 5))`

    bool enable_cast = true;
    if (lane_id < kNumLanes) {
        // Calculate log amin/amax float
        auto amaxmin2 = reinterpret_cast<uint64_t*>(meta_buffer)[lane_id];
        const auto& bf162_amaxmin = reinterpret_cast<__nv_bfloat162*>(&amaxmin2);
        float log_amax[2], log_amin[2];
        #pragma unroll
        for (int i = 0; i < 2; ++ i) { 
            auto amax = static_cast<float>(bf162_amaxmin[i].x);
            auto amin = static_cast<float>(bf162_amaxmin[i].y);
            log_amax[i] = log2f_approx(amax);
            log_amin[i] = amin == 0 ? log_amax[i] - kMinClip : fmaxf(log2f_approx(amin), log_amax[i] - kMinClip);
            enable_cast = enable_cast and log_amax[i] < kLogThreshold and log_amin[i] < log_amax[i];
        }
        shared_log_amax[lane_id] = make_float2(log_amax[0], log_amax[1]);
        shared_log_amin[lane_id] = make_float2(log_amin[0], log_amin[1]);
    }

    const auto& casted = warp_reduce_and<kNumSendUnrolls>(enable_cast) ? 1u << (lane_id / kNumRecvUnrolls): 0u;
    const auto& num_casted_prefix = __popc(warp_reduce_or<kNumRecvUnrolls, true>(casted) & ((1u << (lane_id / kNumRecvUnrolls)) - 1));

    if (lane_id < kNumLanes and lane_id % kNumRecvUnrolls == 0)
        shared_cast_info[lane_id / kNumRecvUnrolls] = (num_casted_prefix << 1) | (casted ? 1u : 0u);
    __syncwarp();
}

template <int kNumRecvUnrolls>
__forceinline__ __device__ void decode_and_accumulate(uint32_t* ld_buffer, float* accum,
                                                      const float& log_amax, const float& log_amin,
                                                      const bool& enable_cast, const float& weight) {
    if (enable_cast) {
        constexpr int kNumBits = 10;
        constexpr int kNumValues = 1 << (kNumBits - 1);

        const auto& step = (log_amax - log_amin) / static_cast<float>(kNumValues - 2);
        auto decode = [=](const uint32_t &encoded, const uint32_t &sign) {
            const auto decoded = encoded == 0 ? .0f : exp2f_approx((encoded - 1) * step + log_amin);
            return sign ? -decoded : decoded;
        };

        EP_STATIC_ASSERT(kNumRecvUnrolls == 2 or kNumRecvUnrolls == 4, "kNumRecvUnrolls == 2 or 4 only");
        #pragma unroll
        for (int i = 0; i < kNumRecvUnrolls / 2; ++ i) {
            uint32_t concat[6];
            concat[0] = ld_buffer[i * 5];
            #pragma unroll
            for (int k = 1; k < 5; ++ k)
                concat[k] = (ld_buffer[i * 5 + k - 1] >> (32 - k * 5)) | (ld_buffer[i * 5 + k] << (k * 5));
            concat[5] = ld_buffer[i * 5 + 4] >> 7;

            const uint32_t& local_signs = ld_buffer[i * 5 + 4] >> 16;
            #pragma unroll
            for (int k = 0; k < 5; ++ k) {
                accum[i * 16 + k * 3 + 0] += decode((concat[k] >>  0) & 0x1ff, (local_signs >> (k * 3 + 0)) & 1) * weight;
                accum[i * 16 + k * 3 + 1] += decode((concat[k] >>  9) & 0x1ff, (local_signs >> (k * 3 + 1)) & 1) * weight;
                accum[i * 16 + k * 3 + 2] += decode((concat[k] >> 18) & 0x1ff, (local_signs >> (k * 3 + 2)) & 1) * weight;
            }
            accum[i * 16 + 15] += decode(concat[5] & 0x1ff, (local_signs >> 15) & 1) * weight;
        }
    } else {
        #pragma unroll
        for (int k = 0; k < kNumRecvUnrolls * 4; ++ k) {
            auto bf16_pack = *reinterpret_cast<__nv_bfloat162*>(ld_buffer + k);
            accum[k * 2 + 0] += static_cast<float>(bf16_pack.x) * weight;
            accum[k * 2 + 1] += static_cast<float>(bf16_pack.y) * weight;
        }
    }
}

template <bool kUseLogFMT, int kHidden, int kNumMaxTopk, int kNumMaxUnrolls, int kEager>
__global__ __launch_bounds__(1024, 1) void
combine(void* combined_x,
        void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
        const void* x, const int64_t* topk_idx, const float* topk_weights,
        const int* per_rank_src_count,
        const int* src_info, const int64_t* layout_range,
        int64_t* combine_wait_recv_cost_stats,
        int* next_clean, int num_next_clean_int,
        int* atomic_clean_flag,
        int num_combined_tokens, int hidden, int num_topk,
        int num_max_dispatch_tokens_per_rank,
        int num_experts, int rank, int num_ranks,
        int num_warp_groups, int num_warps_per_group,
        int phases, bool zero_copy, int combine_round_n) {
    const auto sm_id = __shfl_sync(0xffffffff, static_cast<int>(blockIdx.x), 0);
    const auto num_sms = __shfl_sync(0xffffffff, static_cast<int>(gridDim.x), 0);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = __shfl_sync(0xffffffff, static_cast<int>(blockDim.x), 0);
    const auto warp_id = __shfl_sync(0xffffffff, thread_id / 32, 0), lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr int64_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;

    // Use different unroll factors for send and recv phases
    constexpr int kNumSendUnrolls = kHidden % (32 * 4 * sizeof(int4) / sizeof(nv_bfloat16)) == 0 ? 4 : 2;
    constexpr int kNumRecvUnrolls = 2;
    constexpr int hidden_bf16_int4_pad = align(static_cast<int>(hidden_bf16_int4), 32 * kNumSendUnrolls);
    EP_STATIC_ASSERT(kHidden % (32 * 2 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls <= kNumMaxUnrolls and kNumRecvUnrolls <= kNumMaxUnrolls, "Invalid unrolls");
    EP_STATIC_ASSERT(hidden_bf16_int4 % kNumSendUnrolls == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls >= kNumRecvUnrolls, "Invalid unroll factors");

    // Message package
    EP_STATIC_ASSERT(kHidden % 128 == 0, "Invalid hidden");
    constexpr int kNumDivisions = kHidden / 128;
    constexpr int kNumMetaBytes = kNumDivisions * sizeof(nv_bfloat162);
    constexpr size_t num_bytes_per_slot_v = kHidden * sizeof(nv_bfloat16) + kNumMetaBytes;
    constexpr size_t num_bytes_per_slot = kEager != EAGER_OFF ? EXTEND_FOR_TAG_AND_ALIGN(num_bytes_per_slot_v, AR_MSG_ALIGNMENT) : num_bytes_per_slot_v;
    
    // WARNING!!! Two lines below must be consistent with msg def in config.hpp
    constexpr int kNumPerChannels = 128;
    constexpr int num_scales = kHidden / kNumPerChannels;
    constexpr size_t dispatch_msg_max = sizeof(int4) + std::max(kHidden * sizeof(nv_bfloat16), kHidden + num_scales * sizeof(float));
    constexpr size_t combine_msg_max = num_scales * sizeof(nv_bfloat162) + kHidden * sizeof(nv_bfloat16);
    constexpr size_t msg_distance = kEager != EAGER_OFF ? EXTEND_FOR_TAG_AND_ALIGN(std::max(dispatch_msg_max, combine_msg_max) + sizeof(int), AR_MSG_LONG_ALIGNMENT) : num_bytes_per_slot_v;

    constexpr int kEager_combine = kUseLogFMT ? EAGER_OFF : kEager; // logfmt does not support eager combine
    constexpr int MAX_PAGES_DIV4 = 1;
    constexpr int MAX_PAGES = MAX_PAGES_DIV4 << 2;
    EP_STATIC_ASSERT(MAX_PAGES_DIV4 <= 31, "pages can not be dealt by warp");
    EP_STATIC_ASSERT(kHidden * sizeof(nv_bfloat16) <= MAX_PAGES * PCIE_SEG_LEN, "hidden too large"); // max 4 page, 16K bytes, 8k bf16 hidden
    int4 __tail_tags_int4[MAX_PAGES_DIV4 + 1];
    __tail_tags_int4[MAX_PAGES_DIV4].x = ZTAG(combine_round_n);
    int *__tail_tags = reinterpret_cast<int*>(__tail_tags_int4); // store tag position values

    //constexpr int page_int4 = (PCIE_SEG_LEN - PCIE_TAIL_SZ) / sizeof(int4);
    //constexpr int page_int4_slots = PCIE_SEG_LEN / sizeof(int4);

    EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0, "Invalid vectorization");

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_COMBINE_RECV;

    // Clean up next buffer
    if (sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {
        #pragma unroll
        for (int i = lane_id; i < num_next_clean_int; i += 32)
            next_clean[i] = 0;

        // Notify before executing `int_p`
        __syncwarp();
        if (kEager_combine != EAGER_FULL) {
            if (lane_id == 0)
                atomic_add_release_global(atomic_clean_flag, num_experts);
        }
    }

    // Issue IBGDA sends
    if (responsible_expert_idx < num_experts) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
        const auto layout = kEager != EAGER_FULL ? __ldg(layout_range + local_expert_idx * num_ranks + dst_rank) : 0l;
        const auto layout_bitmap_ptr = reinterpret_cast<const int*>(layout_range) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank + dst_rank * num_max_dispatch_tokens_per_rank; // used in EAGER_FULL
        const int num_tokens_from_this_exp_to_dstrank = kEager != EAGER_FULL ? 0 : __ldg(per_rank_src_count + local_expert_idx * num_ranks + dst_rank); // used in EAGER_FULL

        const auto local_x = static_cast<const int4*>(x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_bf16_int4;
        const auto local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto rdma_send_x_vec = static_cast<uint8_t*>(rdma_send_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot;

        // Unpack layout
        int offset = 0, num_tokens_to_send, begin_idx, end_idx;
        if (kEager != EAGER_FULL) {
            unpack2(layout, num_tokens_to_send, offset);
        }

        begin_idx = sub_warp_id + offset;
        end_idx = kEager == EAGER_FULL ? num_tokens_from_this_exp_to_dstrank : (offset + num_tokens_to_send);
        int token_idx; // position level index

        // TMA stuffs
        constexpr int kNumTMABufferBytes = sizeof(int4) * 32 * kNumSendUnrolls;
        constexpr int kNumStages = 3;
        constexpr int kNumPrefetch = 1;
        EP_STATIC_ASSERT(kNumStages == 3 and kNumPrefetch == 1, "Invalid stages");

        auto smem_ptr = smem_buffer + warp_id * (kNumStages * (kNumTMABufferBytes + 16) + kNumMetaBytes);
        uint32_t tma_phase = 0;
        auto tma_buffers   = PatternVisitor([=](const int& i) { return reinterpret_cast<int4*>(smem_ptr + i * (kNumTMABufferBytes + 16)); });
        auto full_barriers = PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smem_ptr + i * (kNumTMABufferBytes + 16) + kNumTMABufferBytes); });
        auto meta_buffers  = kUseLogFMT ? reinterpret_cast<nv_bfloat162*>(smem_ptr + kNumStages * (kNumTMABufferBytes + 16)) : nullptr;
        EP_STATIC_ASSERT(kNumSendUnrolls * kNumStages <= 12, "TMA buffer size exceed limit");

        // Initialize m-barriers
        if (lane_id < kNumStages) {
            mbarrier_init(full_barriers[lane_id], 1);
            fence_view_async_shared();
            fence_barrier_init();
        }
        __syncwarp();

        constexpr int kNumIters = hidden_bf16_int4_pad / (32 * kNumSendUnrolls);
        auto tma_load_and_arrive = [&](const int& stage_idx, const int4* gmem_ptr, const int& num_bytes) {
            tma_load_1d(tma_buffers[stage_idx], gmem_ptr, full_barriers[stage_idx], num_bytes);
            mbarrier_arrive_and_expect_tx(full_barriers[stage_idx], num_bytes);
        };
        auto get_num_tma_bytes = [&](const int& offset_int4) {
            return min(kNumTMABufferBytes, static_cast<int>((hidden_bf16_int4 - offset_int4) * sizeof(int4)));
        };

        // Issue IBGDA send
        for (int __token_idx = begin_idx; __token_idx < end_idx; __token_idx += num_warps_per_group) { // sequence level index
            if (kEager == EAGER_FULL) {
                token_idx = lane_id == 0 ? __ldg(layout_bitmap_ptr + __token_idx) : 0;
                token_idx = __shfl_sync(0xffffffff, token_idx, 0);
            } else {
                token_idx = __token_idx;
            }
            //EP_DEVICE_ASSERT(token_idx >= 0 && token_idx < num_ranks * num_max_dispatch_tokens_per_rank);
            auto message_idx = __token_idx - offset;

            const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
            const auto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_bytes_per_slot);
            const auto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row);

            // Copy directly to local rank, or copy to buffer and issue RDMA
            const auto src_idx = __shfl_sync(0xffffffff, __ldg(local_src_info + token_idx), 0);
            const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
            const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) + (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * msg_distance;
            const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
            const auto intra_node = (dst_p2p_ptr != 0);
            //if (!intra_node) printf("[rank %d]: an inter-node combine send\n", rank);
            int num_send_bytes = hidden * sizeof(nv_bfloat16);

            // eager mode disable zero copy feature...
            if ((kEager_combine != EAGER_OFF) or (not zero_copy or dst_p2p_ptr != 0)) {
                // Read from `cpy_src_int4_ptr` and copy into `cpy_dst_int4_ptr`
                const auto cpy_src_int4_ptr = (zero_copy && kEager_combine == EAGER_OFF) ? reinterpret_cast<int4*>(buf_ptr) : x_int4;
                const auto cpy_dst_int4_ptr = dst_p2p_ptr == 0 ? reinterpret_cast<int4*>(buf_ptr) : reinterpret_cast<int4*>(dst_p2p_ptr);

                // Prefetch
                if (elect_one_sync(lane_id))
                    tma_load_and_arrive(0, cpy_src_int4_ptr, get_num_tma_bytes(0));
                __syncwarp();

                int tma_offset_bytes = kNumMetaBytes;
                #pragma unroll
                for (int i = lane_id * kNumSendUnrolls, iter_idx = 0; i < hidden_bf16_int4_pad; i += 32 * kNumSendUnrolls, ++ iter_idx) {
                    // Load the next iteration
                    const int& stage_idx = iter_idx % kNumStages;
                    const int& next_stage_idx = (iter_idx + 1) % kNumStages;
                    if (iter_idx + 1 < kNumIters and elect_one_sync(lane_id)) {
                        tma_store_wait<kNumStages - kNumPrefetch - 1>();
                        //const auto& offset_int4 = i + 32 * kNumSendUnrolls;
                        const auto& offset_int4 = (iter_idx + 1) * 32 * kNumSendUnrolls; // why offset_int4 is runtime dynamic?
                        tma_load_and_arrive(next_stage_idx, cpy_src_int4_ptr + offset_int4, get_num_tma_bytes(offset_int4));
                    }
                    __syncwarp();

                    // Wait the current TMA arrival
                    EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
                    mbarrier_wait<true>(full_barriers[stage_idx], tma_phase, stage_idx);
                    if constexpr (kUseLogFMT) {
                        // Cast if possible
                        constexpr int kNumInt4PerDivision = 128 / kNumElemsPerInt4;
                        int num_tma_bytes = logfmt_encode<kNumSendUnrolls>(
                            tma_buffers[stage_idx],
                            // NOTES: only the leader lane will write the result
                            (i % kNumInt4PerDivision == 0) ? meta_buffers + i / kNumInt4PerDivision : nullptr,
                            lane_id);
                        if (elect_one_sync(lane_id))
                            tma_store_1d(tma_buffers[stage_idx], reinterpret_cast<uint8_t*>(cpy_dst_int4_ptr) + tma_offset_bytes, num_tma_bytes);
                        tma_offset_bytes += num_tma_bytes;
                    } else {
                        // BF16 original values
#define COMBINE_SEND_TMA(tma_func, smem_ptr, gmem_ptr, bytes) TMA_AUTO_TAG(tma_func, smem_ptr, gmem_ptr, bytes, cpy_dst_int4_ptr, __tail_tags, combine_round_n, kEager_combine, intra_node)
                        if ((kEager_combine != EAGER_OFF) ? (lane_id == 0) : elect_one_sync(lane_id)) { // force lane 0 to issue tma
                            auto w_offset_int4 = iter_idx * kNumSendUnrolls * 32; // i is dynamic...
                            COMBINE_SEND_TMA(tma_store_1d, tma_buffers[stage_idx], cpy_dst_int4_ptr + w_offset_int4, get_num_tma_bytes(w_offset_int4));
                            //tma_store_1d(tma_buffers[stage_idx], cpy_dst_int4_ptr + w_offset_int4, get_num_tma_bytes(w_offset_int4));
                        }
                    }
                    __syncwarp();
                }

                // Store metadata (min/max values) for LogFMT
                if constexpr (kUseLogFMT) {
                    num_send_bytes = tma_offset_bytes;
                    if (elect_one_sync(lane_id))
                        tma_store_1d(meta_buffers, cpy_dst_int4_ptr, kNumMetaBytes);
                }

                if constexpr (kEager_combine != EAGER_OFF) {
                    // inter_node: write tag save value
                    
                    if (!intra_node) {
                        // reduce or because tags may be stored in different lanes, maybe there is a better way, why just use lane 0 to do tma? If so, reduce can be replaced by shfl_sync
                        __syncwarp();
                        #pragma unroll
                        for (int __pn = 0; __pn < MAX_PAGES; ++__pn) {
                            //__tail_tags[__pn] = warp_reduce_or(__tail_tags[__pn]);
                            __tail_tags[__pn] = __shfl_sync(0xffffffff, __tail_tags[__pn], 0);
                        }

                        //EP_DEVICE_ASSERT(lane_id == get_lane_id());
                        EP_STATIC_ASSERT((kHidden * sizeof(nv_bfloat16)) % sizeof(int4) == 0, "combine message len (no logfmt) shall be a multiple of int4");
                        if (lane_id < MAX_PAGES_DIV4 + 1) {
                            auto target_ptr = cpy_dst_int4_ptr + hidden_bf16_int4 + lane_id;
                            //EP_DEVICE_ASSERT((target_ptr + 1 - cpy_dst_int4_ptr) * sizeof(int4) <= msg_distance);
                            //printf("[rank %d]: exp %d send back rank %d token %d, put stored value and last tag, offset %lu\n", rank, global_expert_idx, dst_rank, src_idx, (target_ptr - cpy_dst_int4_ptr) * sizeof(int4));
                            // if (lane_id == MAX_PAGES_DIV4) EP_DEVICE_ASSERT(__tail_tags_int4[lane_id].x == ZTAG(combine_round_n));
                            // if (lane_id < MAX_PAGES_DIV4) {
                            //     INT_VALUE_NO_NAN(__tail_tags_int4[lane_id].x);
                            //     INT_VALUE_NO_NAN(__tail_tags_int4[lane_id].y);
                            //     INT_VALUE_NO_NAN(__tail_tags_int4[lane_id].z);
                            //     INT_VALUE_NO_NAN(__tail_tags_int4[lane_id].w);
                            // }
                            st_na_release(target_ptr, __tail_tags_int4[lane_id]);
                        }
                        __syncwarp();
                        num_send_bytes = CEIL_ALIGN(num_send_bytes, sizeof(int4)) + (MAX_PAGES_DIV4 + 1) * sizeof(int4); // extra tag store space
                    }
                }

                // Flush all stores
                tma_store_wait();
                __syncwarp();
                if constexpr (kEager_combine != EAGER_OFF) {
                    if (intra_node && lane_id == 0) {
                        // intranode: only write last tag to signal token ready!
                        st_release_sys_global(reinterpret_cast<int*>(cpy_dst_int4_ptr + hidden_bf16_int4 + MAX_PAGES_DIV4), ZTAG(combine_round_n));
                    }
                }
            }

            // Issue RDMA
            // NOTES: for zero-copy mode, we assume the data is already in the send buffer
            if (dst_p2p_ptr == 0)
                nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, num_send_bytes, dst_rank, local_expert_idx, lane_id, message_idx);
        }

        // Put the finishing flag
        EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 16);
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 1), "r"(num_warps_per_group * 32));
        if (sub_warp_id == 1 and lane_id == 0) {
            if (kEager_combine != EAGER_FULL) {
                while (ld_acquire_global(atomic_clean_flag) == 0);
            }
            auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_flag + global_expert_idx);
            auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
            if (dst_p2p_ptr == 0) {
                if (kEager <= EAGER_OFF) {
                    nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), 1, dst_rank, local_expert_idx);
                } else {
                    nvshmemi_ibgda_rma_p(reinterpret_cast<int*>(dst_ptr), 1, dst_rank, local_expert_idx);
                }
            } else {
                st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), 1);
            }
            if (kEager_combine != EAGER_FULL) {
                atomic_add_release_global(atomic_clean_flag, -1);
            }
        }
        __syncwarp();

        // Destroy m-barriers
        if (lane_id < kNumStages) {
            mbarrier_inval(full_barriers[lane_id]);
            fence_view_async_shared();
            fence_barrier_init();
        }
        __syncwarp();
    }

    // Receiving phase
    LOW_LATENCY_COMBINE_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;
    
    if constexpr (kEager_combine != EAGER_FULL) {
        // Wait all ranks to arrive
        if (responsible_expert_idx < num_experts) {
            EP_DEVICE_ASSERT(num_warps_per_group > 1);
            if (sub_warp_id == 0 and lane_id == 0) {
                auto start_time = clock64();
                while (ld_acquire_sys_global(rdma_recv_flag + responsible_expert_idx) == 0);
                auto wait_recv_cost = clock64() - start_time;
                if (combine_wait_recv_cost_stats != nullptr) {
                    const auto& src_rank = responsible_expert_idx / num_local_experts;
                    atomicAdd(reinterpret_cast<unsigned long long*>(combine_wait_recv_cost_stats + src_rank), wait_recv_cost);
                }
            }
        }
        cg::this_grid().sync();
    } else {
        __syncthreads(); // two types of recv warp must be synchronized
    }
    

    // Reassign warp groups
    constexpr int kMaxNumGroups = 2;
    const int num_decode_warps = hidden_bf16_int4_pad / (kNumRecvUnrolls * 32);
    const int num_groups = min(kMaxNumGroups, (num_threads / 32) / (num_decode_warps + 1));
    const int decode_warp_idx = __shfl_sync(0xffffffff, warp_id % (num_decode_warps + 1), 0);
    const int group_idx = __shfl_sync(0xffffffff, warp_id / (num_decode_warps + 1), 0);
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0, "Invalid vectorization");
    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT(num_groups > 0);

    if (group_idx < num_groups) {
        constexpr int kNumStages = 3;
        constexpr int kNumTMABufferBytes = 16 * 2 + kHidden * 2;
        constexpr int kNumBF16PerWarpBytes = 32 * kNumRecvUnrolls * kNumElemsPerInt4 * 2;
        constexpr int kNumLogFMTPerWarpBytes = kNumBF16PerWarpBytes / 16 * 10;
        constexpr int kNumDivisionBytes = kNumDivisions * sizeof(uint32_t);
        constexpr int kNumBytesPerGroup = kNumStages * kNumTMABufferBytes + kHidden * 2 + kNumStages * kNumDivisionBytes * 3;

        __shared__ int4 ld_tags_int4[kMaxNumGroups][kNumStages][MAX_PAGES_DIV4];

        // Reallocate shared memory
        const auto smem_group_buffer = smem_buffer + kNumBytesPerGroup * group_idx;
        auto full_barriers  = PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smem_group_buffer + i * kNumTMABufferBytes); });
        auto empty_barriers = PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smem_group_buffer + i * kNumTMABufferBytes + 8); });
        auto tma_ld_buffers = PatternVisitor([=](const int& i) { return reinterpret_cast<uint8_t* >(smem_group_buffer + i * kNumTMABufferBytes + 16); });
        auto tma_st_buffers = PatternVisitor([=](const int& i) { return reinterpret_cast<uint32_t*>(smem_group_buffer + kNumStages * kNumTMABufferBytes + i * kNumBF16PerWarpBytes); });

        // Redundant when logfmt is disabled
        const auto smem_group_ptr = smem_group_buffer + kNumStages * kNumTMABufferBytes + kHidden * 2;
        auto log_amax_buffers  = PatternVisitor([=](const int& i) { return reinterpret_cast<float*>(smem_group_ptr + i * kNumDivisionBytes); });
        auto log_amin_buffers  = PatternVisitor([=](const int& i) { return reinterpret_cast<float*>(smem_group_ptr + kNumStages * kNumDivisionBytes + i * kNumDivisionBytes); });
        auto cast_info_buffers = PatternVisitor([=](const int& i) { return reinterpret_cast<int*>  (smem_group_ptr + kNumStages * kNumDivisionBytes * 2 + i * kNumDivisionBytes); });

        uint32_t tma_phase = 0;
        EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
        if (decode_warp_idx == num_decode_warps)
            tma_phase = (1 << kNumStages) - 1;

        // Initialize m-barriers
        if (decode_warp_idx == num_decode_warps and lane_id < kNumStages) {
            mbarrier_init(full_barriers[lane_id], 1);
            mbarrier_init(empty_barriers[lane_id], num_decode_warps);
        }
        asm volatile("bar.sync %0, %1;" :: "r"(group_idx + 1), "r"((num_decode_warps + 1) * 32));

        int stage_idx = 0, topk_idx_by_lane = 0;
        EP_STATIC_ASSERT(kNumMaxTopk <= 32, "Invalid number of topks");
        if (decode_warp_idx == num_decode_warps) {
            // TMA load warp
            //if (sm_id == 0 && rank == 0 && group_idx == 0 && lane_id == 0) printf("tma load warp working\n");
            for (int token_idx = sm_id + num_sms * group_idx; token_idx < num_combined_tokens; token_idx += num_sms * num_groups) {
                if (lane_id < num_topk)
                    topk_idx_by_lane = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id));
                for (int i = 0; i < num_topk; ++ i) {
                    int topk_idx_reg = __shfl_sync(0xffffffff, topk_idx_by_lane, i);
                    if (topk_idx_reg < 0)
                        continue;
                    const int src_rank = topk_idx_reg / num_local_experts;
                    const auto intra_node = (src_rank >> 3) == (rank >> 3);
                    //if (!intra_node) printf("[rank %d]: an inter-node combine recv\n", rank);
                    //if (sm_id == 0 && rank == 0 && group_idx == 0 && lane_id == 0) printf("tma load warp waiting on mbarrier\n");
                    mbarrier_wait<true>(empty_barriers[stage_idx], tma_phase, stage_idx);
                    auto buffer = static_cast<uint8_t*>(rdma_recv_x) + (topk_idx_reg * num_max_dispatch_tokens_per_rank + token_idx) * msg_distance;
                    //if (sm_id == 0 && rank == 0 && group_idx == 0 && lane_id == 0) printf("tma load warp pass mbarrier\n");
                    if constexpr (kEager_combine != EAGER_OFF) {
                        constexpr int pages = CEIL_DIV(kHidden * sizeof(nv_bfloat16), PCIE_SEG_LEN);
                        constexpr int last_page_part = (kHidden * sizeof(nv_bfloat16)) & PCIE_SEG_LEN_MASK;
                        constexpr int full_last_page = last_page_part == 0 ? 1 : 0;
                        if (intra_node) {
                            // intra node: just check tail tag
                            if (lane_id == 0) {
                                int *__check_ptr = reinterpret_cast<int*>(buffer + kHidden * sizeof(nv_bfloat16) + MAX_PAGES_DIV4 * sizeof(int4));
                                while (ld_acquire_sys_global(__check_ptr) != ZTAG(combine_round_n));
                            }
                        } else {
                            // inter node: check tags of all pages
                            #pragma unroll
                            for (int __pn = lane_id; __pn < pages + full_last_page; __pn += 32) {
                                int *__check_ptr = reinterpret_cast<int*>(buffer + (__pn << PCIE_SEG_LEN_LOG) + ((__pn == pages + full_last_page - 1) ? (last_page_part + MAX_PAGES_DIV4 * sizeof(int4)) : (PCIE_SEG_LEN - PCIE_TAIL_SZ)));
                                //EP_DEVICE_ASSERT(reinterpret_cast<uint8_t*>(__check_ptr) - buffer + sizeof(int) <= msg_distance);
                                //printf("[rank %d]: token %d topk %d from exp %d at rank %d, check offset %lu\n", rank, token_idx, i, topk_idx_reg, src_rank, reinterpret_cast<uint8_t*>(__check_ptr) - buffer);
                                while (ld_acquire_sys_global(__check_ptr) != ZTAG(combine_round_n));
                                //printf("[rank %d]: token %d topk %d from exp %d at rank %d, check offset %lu, OK\n", rank, token_idx, i, topk_idx_reg, src_rank, reinterpret_cast<uint8_t*>(__check_ptr) - buffer);
                            }
                        }
                        __syncwarp();
                        if (!intra_node) {
                            if (lane_id < MAX_PAGES_DIV4) {
                                auto tag_pos = buffer + sizeof(nv_bfloat16) * kHidden + lane_id * sizeof(int4);
                                //EP_DEVICE_ASSERT(tag_pos - buffer + sizeof(int4) <= msg_distance);
                                st_release_cta(&ld_tags_int4[group_idx][stage_idx][lane_id], ld_nc_global(reinterpret_cast<int4*>(tag_pos)));
                            }
                        }
                        __syncwarp();
                    }
                    if constexpr (kUseLogFMT) {
                        logfmt_check_amaxmin<kNumDivisions / 2, kNumSendUnrolls, kNumRecvUnrolls>(
                            buffer, reinterpret_cast<float2*>(log_amax_buffers[stage_idx]),
                            reinterpret_cast<float2*>(log_amin_buffers[stage_idx]), cast_info_buffers[stage_idx], lane_id);
                    }
                    //if (sm_id == 0 && rank == 0 && group_idx == 0 && lane_id == 0) printf("tma load warp launching tma load\n");
                    if (elect_one_sync(lane_id)) {
                        int num_casted = 0;
                        if constexpr (kUseLogFMT) {
                            const auto& info = cast_info_buffers[stage_idx][num_decode_warps - 1];
                            num_casted = (info >> 1) + (info & 1);
                        }
                        int num_tma_bytes = num_casted * kNumLogFMTPerWarpBytes + (num_decode_warps - num_casted) * kNumBF16PerWarpBytes;
                        tma_load_1d(tma_ld_buffers[stage_idx], buffer + (kUseLogFMT ? kNumMetaBytes : 0), full_barriers[stage_idx], num_tma_bytes);
                        mbarrier_arrive_and_expect_tx(full_barriers[stage_idx], num_tma_bytes);
                    }
                    //if (sm_id == 0 && rank == 0 && group_idx == 0 && lane_id == 0) printf("tma load warp started tma load\n");
                    // if constexpr (kEager != EAGER_OFF) {
                    //     // inter node: tag postion shall be restored, load them first
                    //     if (!intra_node && lane_id < MAX_PAGES_DIV4) {
                    //         auto tag_pos = buffer + sizeof(nv_bfloat16) * kHidden + lane_id * sizeof(int4);
                    //         EP_DEVICE_ASSERT(tag_pos - buffer + sizeof(int4) <= msg_distance);
                    //         ld_tags_int4[group_idx][stage_idx][lane_id] = ld_nc_global(reinterpret_cast<int4*>(tag_pos));
                    //     }
                    // }
                    __syncwarp();
                    stage_idx = (stage_idx + 1) % kNumStages;
                    //if (sm_id == 0 && rank == 0 && group_idx == 0 && lane_id == 0) printf("tma load warp loop end\n");
                }
            }
        } else {
            // Reduction warps
            //if (sm_id == 0 && rank == 0 && group_idx == 0 && lane_id == 0 && decode_warp_idx == 0) printf("reduce warp working\n");
            float topk_weights_by_lane;
            for (int token_idx = sm_id + num_sms * group_idx; token_idx < num_combined_tokens; token_idx += num_sms * num_groups) {
                if (lane_id < num_topk) {
                    topk_idx_by_lane = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id));
                    topk_weights_by_lane = __ldg(topk_weights + token_idx * num_topk + lane_id);
                }
                __syncwarp();
                //if (sm_id == 0 && rank == 0 && group_idx == 0 && lane_id == 0 && decode_warp_idx == 0) printf("reduce warp got topk info\n");
                float combined_values[kNumElemsPerInt4 * kNumRecvUnrolls] = {0.0f};
                for (int i = 0; i < num_topk; ++ i) {
                    const auto topk_idx = __shfl_sync(0xffffffff, topk_idx_by_lane, i);
                    if (topk_idx < 0)
                        continue;
                    const auto& topk_weight = __shfl_sync(0xffffffff, topk_weights_by_lane, i);
                    const auto intra_node = ((topk_idx / num_local_experts) >> 3) == (rank >> 3);
                    //if (sm_id == 0 && rank == 0 && group_idx == 0 && lane_id == 0 && decode_warp_idx == 0) printf("reduce warp wait on barrier\n");
                    mbarrier_wait<true>(full_barriers[stage_idx], tma_phase, stage_idx);
                    //if (sm_id == 0 && rank == 0 && group_idx == 0 && lane_id == 0 && decode_warp_idx == 0) printf("reduce warp pass barrier\n");
                    if constexpr (kEager_combine != EAGER_OFF) {
                        auto buffer = static_cast<uint8_t*>(rdma_recv_x) + (topk_idx * num_max_dispatch_tokens_per_rank + token_idx) * msg_distance;
                        constexpr int pages = (kHidden * sizeof(nv_bfloat16)) >> PCIE_SEG_LEN_LOG;
                        if (!intra_node) {
                            if (decode_warp_idx == 0 && lane_id < pages) {
                                //EP_DEVICE_ASSERT((lane_id << PCIE_SEG_LEN_LOG) + (PCIE_SEG_LEN - PCIE_TAIL_SZ) < kHidden * sizeof(nv_bfloat16) - sizeof(int));
                                // int space;
                                // nv_bfloat16* space_ptr = reinterpret_cast<nv_bfloat16*>(&space);
                                // space_ptr[0] = static_cast<nv_bfloat16>(rank - 128);
                                // space_ptr[1] = space_ptr[0];
                                //EP_DEVICE_ASSERT(reinterpret_cast<int*>(ld_tags_int4[group_idx][stage_idx])[lane_id] == space);
                                //printf("[rank %d]: debug: wriring shmem\n", rank);
                                int save_value = ld_nc_global(reinterpret_cast<int*>(buffer + kHidden * sizeof(nv_bfloat16) + lane_id * sizeof(int))); // reinterpret_cast<int*>(ld_tags_int4[group_idx][stage_idx])[lane_id]
                                //INT_VALUE_NO_NAN(save_value);
                                st_release_cta(reinterpret_cast<int*>(tma_ld_buffers[stage_idx] + (lane_id << PCIE_SEG_LEN_LOG) + (PCIE_SEG_LEN - PCIE_TAIL_SZ)), save_value);
                            }
                            asm volatile("bar.sync %0, %1;" :: "r"(group_idx + 2), "r"(num_decode_warps * 32));
                        }
                    }
                    //if (sm_id == 0 && rank == 0 && group_idx == 0 && lane_id == 0 && decode_warp_idx == 0) printf("reduce warp begin reduce\n");
                    if constexpr (kUseLogFMT) {
                        const auto& info = cast_info_buffers[stage_idx][decode_warp_idx];
                        bool enable_cast = info & 1;
                        int num_casted_prefix = info >> 1;
                        int tma_offset = kNumLogFMTPerWarpBytes * num_casted_prefix + kNumBF16PerWarpBytes * (decode_warp_idx - num_casted_prefix);
                        int division_idx = decode_warp_idx * (kNumRecvUnrolls * 2) + lane_id * kNumRecvUnrolls / 16;
                        decode_and_accumulate<kNumRecvUnrolls>(
                            reinterpret_cast<uint32_t*>(tma_ld_buffers[stage_idx] + tma_offset + (enable_cast ? kNumLogFMTPerWarpBytes : kNumBF16PerWarpBytes) / 32 * lane_id),
                            combined_values, log_amax_buffers[stage_idx][division_idx], log_amin_buffers[stage_idx][division_idx], enable_cast, topk_weight
                        );
                    } else {
                        int tma_offset = kNumBF16PerWarpBytes * decode_warp_idx;
                        decode_and_accumulate<kNumRecvUnrolls>(
                            reinterpret_cast<uint32_t*>(tma_ld_buffers[stage_idx] + tma_offset + kNumBF16PerWarpBytes / 32 * lane_id),
                            combined_values, 0, 0, false, topk_weight
                        );
                    }
                    //if (sm_id == 0 && rank == 0 && group_idx == 0 && lane_id == 0 && decode_warp_idx == 0) printf("reduce warp finish reduce\n");
                    if (elect_one_sync(lane_id))
                        mbarrier_arrive(empty_barriers[stage_idx]);
                    stage_idx = (stage_idx + 1) % kNumStages;
                }
                //asm volatile("bar.sync %0, %1;" :: "r"(group_idx + 2), "r"(num_decode_warps * 32));
                //if (sm_id == 0 && rank == 0 && group_idx == 0 && lane_id == 0 && decode_warp_idx == 0) printf("reduce warp finish all reduce, waiting for tma store done\n");

                tma_store_wait<0>();

                //if (sm_id == 0 && rank == 0 && group_idx == 0 && lane_id == 0 && decode_warp_idx == 0) printf("reduce warp finish all reduce, load reduce result to share mem\n");
                #pragma unroll
                for (int k = 0; k < kNumRecvUnrolls * 4; ++ k) {
                    auto combined_pack = __nv_bfloat162(combined_values[k * 2], combined_values[k * 2 + 1]);
                    tma_st_buffers[decode_warp_idx][kNumRecvUnrolls * 4 * lane_id + k] = *reinterpret_cast<uint32_t*>(&combined_pack);
                }
                tma_store_fence();
                //if (sm_id == 0 && rank == 0 && group_idx == 0 && lane_id == 0 && decode_warp_idx == 0) printf("reduce warp finish all reduce, start tma store\n");
                if (elect_one_sync(lane_id)) {
                    tma_store_1d(tma_st_buffers[decode_warp_idx],
                                 static_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4 + decode_warp_idx * kNumRecvUnrolls * 32,
                                 kNumBF16PerWarpBytes);
                }
                __syncwarp();
                //if (sm_id == 0 && rank == 0 && group_idx == 0 && lane_id == 0 && decode_warp_idx == 0) printf("reduce warp finish a token\n");
            }
        }

        // Flush all stores
        tma_store_wait<0>();
    }
}

void combine(void* combined_x,
             void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
             const void* x, const int64_t* topk_idx, const float* topk_weights,
             const int* per_rank_src_count,
             const int* src_info, const int64_t* layout_range,
             int64_t* combine_wait_recv_cost_stats,
             int* next_clean, int num_next_clean_int,
             int num_combined_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
             int num_topk, int num_experts, int rank, int num_ranks,
             bool use_logfmt,
             void* workspace, int num_device_sms,
             cudaStream_t stream, int phases, bool zero_copy,
             int combine_round_n, const int eager_opt) {
    constexpr int kNumMaxTopk = 9;
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = 32 / num_warp_groups;
    const int num_recv_per_sm = ceil_div(num_combined_tokens, num_device_sms);
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0 and num_recv_per_sm > 0);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms = max(ceil_div(num_experts, num_warp_groups), ceil_div(num_combined_tokens, num_recv_per_sm));

    // Check workspace
    auto atomic_clean_flag = static_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

    // Online cast cannot use zero-copy
    EP_HOST_ASSERT(not (zero_copy and use_logfmt));

    constexpr int kNumStages = 3;
    constexpr int kNumMaxUnrolls = 4;
    constexpr int kMaxNumGroups = 2;

    // Send buffer size
    const int num_meta_bytes = hidden / 128 * 4;
    const int num_send_tma_bytes = 32 * sizeof(int4) * kNumMaxUnrolls + 16;
    const int smem_send_size = num_warps * (kNumStages * num_send_tma_bytes + num_meta_bytes);

    // Receive buffer size
    const int num_recv_tma_bytes = 16 + hidden * 2;
    const int smem_recv_size = kMaxNumGroups * (kNumStages * num_recv_tma_bytes + hidden * 2 + kNumStages * num_meta_bytes * 3);

    // Total requirement
    const int smem_size = max(smem_send_size, smem_recv_size);

#define COMBINE_LAUNCH_CASE(eager_opt, hidden) { \
auto combine_func = use_logfmt ? \
    combine<true, hidden, kNumMaxTopk, kNumMaxUnrolls, EAGER_OFF> : \
    combine<false, hidden, kNumMaxTopk, kNumMaxUnrolls, eager_opt>; \
SET_SHARED_MEMORY_FOR_TMA(combine_func); \
LAUNCH_KERNEL(&cfg, combine_func, \
              combined_x, \
              rdma_recv_x, rdma_recv_flag, rdma_send_x, \
              x, topk_idx, topk_weights, per_rank_src_count, src_info, layout_range, \
              combine_wait_recv_cost_stats, \
              next_clean, num_next_clean_int, \
              atomic_clean_flag, \
              num_combined_tokens, hidden, num_topk, \
              num_max_dispatch_tokens_per_rank, \
              num_experts, rank, num_ranks, \
              num_warp_groups, num_warps_per_group, \
              phases, zero_copy, combine_round_n); } break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);

    #define EAGER_COMBINE_LAUNCH_CASE(hidden) SWITCH_EAGER(COMBINE_LAUNCH_CASE, hidden)

    SWITCH_HIDDEN(EAGER_COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

} // namespace internode_ll

} // namespace deep_ep
