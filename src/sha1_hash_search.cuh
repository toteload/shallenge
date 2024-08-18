#pragma once

#include "common.hpp"
#include "jobgenerator.hpp"

constexpr u32 ALPHABET_SIZE     = 64;
constexpr u32 SEARCH_BLOCK_SIZE = 3;
constexpr u32 SHA256_STATE_SIZE = 8;

extern u8 alphabet[ALPHABET_SIZE];

           __host__ void initialize_cuda_constants();
           __host__ void write_payload(JobDescription const &job, char const *header, u32 header_len, u8 *payload);
__device__ __host__ bool is_better_hash(u32 const best[5], u32 const candidate[5]);

           __host__ void sha256_host(const u8 block[64], u32 state[SHA256_STATE_SIZE]);

__global__ void search_block(const u8 *base_payload, const u32 *idx, u8 *base_has_improvement, u32 best_block, u32 zero_block_count);
