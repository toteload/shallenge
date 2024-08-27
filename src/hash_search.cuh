#pragma once

#include "common.hpp"

constexpr u32 SEARCH_BLOCK_SIZE = 3;
constexpr u32 SHA256_STATE_SIZE = 8;

           __host__ void initialize_cuda_constants();
__device__ __host__ bool is_better_hash(u32 const best[5], u32 const candidate[5]);

           __host__ void sha256_host(const u8 block[64], u32 state[SHA256_STATE_SIZE]);

__global__ void search_block(const u8 payload[64], const u32 *idx, u32 *out);
