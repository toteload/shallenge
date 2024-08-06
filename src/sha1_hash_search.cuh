#pragma once

#include "common.hpp"
#include "jobgenerator.hpp"

constexpr u32 ALPHABET_SIZE     = 64;
constexpr u32 SEARCH_BLOCK_SIZE = 3;

extern u8 alphabet[ALPHABET_SIZE];

           __host__ void initialize_cuda_constants();
           __host__ void write_payload(JobDescription const &job, char const *header, u32 header_len, u8 *payload);
__device__ __host__ bool is_better_hash(u32 best[5], u32 candidate[5]);

__global__ void search_block(const u8 payload[64], const u32 *idx, u32 *out);
