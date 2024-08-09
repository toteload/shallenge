#include "common.hpp"
#include "jobgenerator.hpp"
#include "sha1_hash_search.cuh"

#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <vector>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    initialize_cuda_constants();

    int grid_size, block_size;
    cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, search_block, 0, 0);

    printf("grid: %u, block: %u\n", grid_size, block_size);

    cudaStream_t stream[4];
    for (u32 i = 0; i < 4; i++) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream[i]));
    }

    char const *header = "toteload/davidbos+dot+me/";
    u32 header_len = strlen(header);

    JobGenerator generator(55 - header_len);

    std::vector<JobDescription> jobs;
    jobs.resize(block_size * grid_size);

    uint8_t *payload;
    CHECK_CUDA_ERROR(cudaMallocManaged(&payload, grid_size * block_size * 64));

    uint32_t *out;
    CHECK_CUDA_ERROR(cudaMallocManaged(&out, block_size * grid_size * 5 * sizeof(u32)));

    u32 *idx;
    CHECK_CUDA_ERROR(cudaMallocManaged(&idx, block_size * grid_size * 3 * sizeof(u32)));

    u32 best_hash[5] = { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, };

    auto start = std::chrono::high_resolution_clock::now();

    for (u32 i = 0; i < 100; i++) {
        for (u32 i = 0; i < jobs.size(); i++) {
            generator.next(jobs[i]);
        }

        for (u32 i = 0; i < jobs.size(); i++) {
            write_payload(jobs[i], header, header_len, payload + i * 64);
        }

        for (u32 i = 0; i < jobs.size(); i++) {
            idx[i*3+0] = jobs[i].search_idxs[0];
            idx[i*3+1] = jobs[i].search_idxs[1];
            idx[i*3+2] = jobs[i].search_idxs[2];
        }

        search_block<<<grid_size, block_size>>>(payload, idx, out);

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        for (int i = 0; i < block_size * grid_size; i++) {
            u32 *candidate = out + i * 5;
            if (is_better_hash(best_hash, candidate)) {
                memcpy(best_hash, candidate, 20);
            }
        }

        for (int i = 0; i < 5; i++) {
            printf("%08x ", best_hash[i]);
        }

        puts("");
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    printf("duration %f s\n", d.count());
    printf("%f MH/s\n", (double)(64*64*64) * 100 * block_size * grid_size / 1'000'000.0 / d.count());

    cudaFree(payload);
    cudaFree(out);

    return 0;
}

