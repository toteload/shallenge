#pragma once

#include "common.hpp"

constexpr u32 SEARCH_BLOCK_SIZE = 3;
constexpr u32 SHA256_STATE_SIZE = 8;

           __host__ void initialize_cuda_constants();
__device__ __host__ bool is_better_hash(u32 const best[5], u32 const candidate[5]);
__device__ __host__ bool is_better_hash_head(u32 const best[2], u32 const candidate[2]);

           __host__ void sha256(const u8 block[64], u32 state[SHA256_STATE_SIZE]);
           __host__ void sha256_big_endian(const u8 block[64], u32 state[SHA256_STATE_SIZE]);

__device__ __host__ void sha256_l55_prepare(u32 m[16], u32 reg[8]);
__device__ __host__ void sha256_l55(u32 m[16], u32 reg[8], u32 state_head_out[2]);

__global__          void search_block(const u8 *base_payload, u32 *base_out);
