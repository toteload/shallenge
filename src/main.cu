#include "common.hpp"
#include "jobgenerator.hpp"
#include "hash_search.cuh"

#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include <iostream>
#include <chrono>
#include <vector>

#define CHECK_CUDA_ERROR(val) check((val), __FILE__, __LINE__)
void check(cudaError_t err, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

struct Context {
    struct {
        u8  *payload;
        u32 *out;
    } host;

    struct {
        u8  *payload;
        u32 *out;
    } device;

    int grid_size;
    int block_size;

    cudaStream_t stream;
    cudaEvent_t  event;
};

void search(Context const &ctx) {
    u32 const payload_buffer_size = ctx.grid_size * ctx.block_size * 64;
    u32 const out_buffer_size     = ctx.grid_size * ctx.block_size * 2 * sizeof(u32);

    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(
            ctx.device.payload,
            ctx.host.payload,
            payload_buffer_size,
            cudaMemcpyHostToDevice,
            ctx.stream
        )
    );

    search_block<<<ctx.grid_size, ctx.block_size, 0, ctx.stream>>>(
        ctx.device.payload,
        ctx.device.out
    );

    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(
            ctx.host.out,
            ctx.device.out,
            out_buffer_size,
            cudaMemcpyDeviceToHost,
            ctx.stream
        )
    );

    CHECK_CUDA_ERROR(cudaEventRecord(ctx.event, ctx.stream));
}

void find_best_payload(u8 const search_payload[64], u8 payload_out[64], u32 hash_out[8]) {
    u32 best[8] = {
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 
    };

    u8 best_payload[64];

    for (u32 i0 = 0; i0 < ALPHABET_SIZE; i0++) {
        for (u32 i1 = 0; i1 < ALPHABET_SIZE; i1++) {
            for (u32 i2 = 0; i2 < ALPHABET_SIZE; i2++) {

                u8 payload[64];
                memcpy(payload, search_payload, 64);

                payload[53] = alphabet[i0];
                payload[54] = alphabet[i1];
                payload[55] = alphabet[i2];

                u32 candidate[SHA256_STATE_SIZE] = { 
                    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
                };

                sha256_big_endian(payload, candidate);

                if (is_better_hash(best, candidate)) {
                    memcpy(best, candidate, 32);
                    memcpy(best_payload, payload, 64);
                }
            } 
        } 
    }

    memcpy(hash_out, best, 32);
    memcpy(payload_out, best_payload, 64);
}

int main() {
    initialize_cuda_constants();

    int grid_size, block_size;
    cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, search_block, 0, 0);

    printf("Grid size: %u | Block size: %u\n", grid_size, block_size);

    char const *header = "toteload/davidbos+dot+me/";

    LongJobGenerator generator(header);

    cudaStream_t stream[2];
    for (u32 i = 0; i < 2; i++) {
        CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));
    }

    cudaEvent_t event[2];
    for (u32 i = 0; i < 2; i++) {
        CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&event[i], cudaEventDisableTiming | cudaEventBlockingSync));
    }

    u32 const payload_buffer_size = grid_size * block_size * 64;
    u32 const out_buffer_length   = grid_size * block_size * 2;
    u32 const out_buffer_size     = out_buffer_length * sizeof(u32);

    u8  *payload_host;
    u32 *out_host;

    u8  *payload_device;
    u32 *out_device;

    CHECK_CUDA_ERROR(cudaMallocHost(&payload_host, 2 * payload_buffer_size));
    CHECK_CUDA_ERROR(cudaMallocHost(&out_host,     2 * out_buffer_size));

    CHECK_CUDA_ERROR(cudaMalloc(&payload_device, 2 * payload_buffer_size));
    CHECK_CUDA_ERROR(cudaMalloc(&out_device,     2 * out_buffer_size));

    u32 buf_idx = 0;

    for (u32 i = 0; i < grid_size * block_size; i++) {
        generator.next(payload_host + i * 64);
    }

    search({
        .host = {
            .payload = payload_host,
            .out     = out_host,
        },
        .device = {
            .payload = payload_device,
            .out     = out_device,
        },

        .grid_size  = grid_size,
        .block_size = block_size,

        .stream = stream[0],
        .event  = event[0],
    });

    u8  best_payload[64];
    u32 best_hash[SHA256_STATE_SIZE] = { 
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 
    };

    while (true) {
        auto start = std::chrono::high_resolution_clock::now();

        u32 next_buf_idx = 1 - buf_idx;

        for (u32 i = 0; i < grid_size * block_size; i++) {
            generator.next(payload_host + next_buf_idx * payload_buffer_size + i * 64);
        }

        search({
            .host = {
                .payload = payload_host + next_buf_idx * payload_buffer_size,
                .out     = out_host     + next_buf_idx * out_buffer_length,
            },
            .device = {
                .payload = payload_device + next_buf_idx * payload_buffer_size,
                .out     = out_device     + next_buf_idx * out_buffer_length,
            },

            .grid_size  = grid_size,
            .block_size = block_size,

            .stream = stream[next_buf_idx],
            .event  = event[next_buf_idx],
        });

        CHECK_CUDA_ERROR(cudaEventSynchronize(event[buf_idx]));

        u32 const *out = out_host + buf_idx * out_buffer_length;

        for (int i = 0; i < block_size * grid_size; i++) {
            u32 const *candidate = out + i * 2;

            if (is_better_hash_head(best_hash, candidate)) {
                u8 *search_payload = payload_host + buf_idx * payload_buffer_size + i * 64;
                find_best_payload(search_payload, best_payload, best_hash);

                for (u32 i = 0; i < 16; i++) {
                    std::swap(best_payload[i*4+0], best_payload[i*4+3]);
                    std::swap(best_payload[i*4+1], best_payload[i*4+2]);
                }

                printf("NEW BEST! %.55s - %08x %08x %08x %08x %08x %08x %08x %08x\n",
                    best_payload,

                    best_hash[0],
                    best_hash[1],
                    best_hash[2],
                    best_hash[3],
                    best_hash[4],
                    best_hash[5],
                    best_hash[6],
                    best_hash[7]
                );
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        double hashrate = (double)(64*64*64) * grid_size * block_size / 1'000'000'000.0 / d.count();
        printf("%3.3f GH/s | Last = ???\n", hashrate);

        buf_idx = next_buf_idx;
    }

    return 0;
}

