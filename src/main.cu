#include "common.hpp"
#include "jobgenerator.hpp"
#include "sha1_hash_search.cuh"

#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include <iostream>
#include <chrono>
#include <vector>

static_assert(sizeof(u64) == 8);

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
        u32 *search_index;
        u8  *has_improvement;
    } host;

    struct {
        u8  *payload;
        u32 *search_index;
        u8  *has_improvement;
    } device;

    int grid_size;
    int block_size;

    u32 zero_block_count;
    u32 best_block;

    cudaStream_t stream;
    cudaEvent_t  event;
};

void search(Context const &ctx, std::vector<JobDescription> &jobs, char const *header, u32 header_len) {
    u32 const payload_buffer_size      = ctx.grid_size * ctx.block_size * 64;
    u32 const search_index_buffer_size = ctx.grid_size * ctx.block_size * 12;
    u32 const has_improvement_buffer_size          = ctx.grid_size * ctx.block_size;

    for (u32 i = 0; i < jobs.size(); i++) {
        write_payload(jobs[i], header, header_len, ctx.host.payload + i * 64);
    }

    for (u32 i = 0; i < jobs.size(); i++) {
        ctx.host.search_index[i*3+0] = header_len + jobs[i].search_idxs[0];
        ctx.host.search_index[i*3+1] = header_len + jobs[i].search_idxs[1];
        ctx.host.search_index[i*3+2] = header_len + jobs[i].search_idxs[2];
    }

    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(
            ctx.device.payload,
            ctx.host.payload,
            payload_buffer_size,
            cudaMemcpyHostToDevice,
            ctx.stream
        )
    );

    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(
            ctx.device.search_index,
            ctx.host.search_index,
            search_index_buffer_size,
            cudaMemcpyHostToDevice,
            ctx.stream
        )
    );

    search_block<<<ctx.grid_size, ctx.block_size, 0, ctx.stream>>>(
        ctx.device.payload,
        ctx.device.search_index,
        ctx.device.has_improvement,
        ctx.best_block,
        ctx.zero_block_count
    );

    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(
            ctx.host.has_improvement,
            ctx.device.has_improvement,
            has_improvement_buffer_size,
            cudaMemcpyDeviceToHost,
            ctx.stream
        )
    );

    CHECK_CUDA_ERROR(cudaEventRecord(ctx.event, ctx.stream));
}

void find_best_payload(
    JobDescription const &job,
    char const *header,
    u32 header_len,
    u8 out_payload[64],
    u32 out_best_hash[SHA256_STATE_SIZE])
{
    u8 payload[64];
    write_payload(job, header, header_len, payload);

    u32 best[SHA256_STATE_SIZE] = {
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    };

    for (u32 i0 = 0; i0 < ALPHABET_SIZE; i0++) { payload[header_len + job.search_idxs[0]] = alphabet[i0];
    for (u32 i1 = 0; i1 < ALPHABET_SIZE; i1++) { payload[header_len + job.search_idxs[1]] = alphabet[i1];
    for (u32 i2 = 0; i2 < ALPHABET_SIZE; i2++) { payload[header_len + job.search_idxs[2]] = alphabet[i2];
        u32 candidate[SHA256_STATE_SIZE] = { 
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
        };

        sha256_host(payload, candidate);

        if (is_better_hash(best, candidate)) {
            memcpy(best, candidate, 32);
            memcpy(out_payload, payload, 64);
        }
    } } }

    memcpy(out_best_hash, best, 32);
}

int main() {
    initialize_cuda_constants();

    int grid_size, block_size;
    cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, search_block, 0, 0);

    printf("Grid size: %u | Block size: %u\n", grid_size, block_size);

    char const *header = "toteload/davidbos+dot+me/";
    u32 header_len = strlen(header);

    JobGenerator generator(55 - header_len);

#if 0
    generator.it_len          = 7;
    generator.search_idxs[0]  = 3; 
    generator.search_idxs[1]  = 1; 
    generator.search_idxs[2]  = 0; 
    generator.product_idx     = 16109310;
    generator.product_idx_max = 64 * 64 * 64 * 64;
#endif

    std::vector<JobDescription> jobs[2];
    jobs[0].resize(block_size * grid_size);
    jobs[1].resize(block_size * grid_size);

    cudaStream_t stream[2];
    for (u32 i = 0; i < 2; i++) {
        CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));
    }

    cudaEvent_t event[2];
    for (u32 i = 0; i < 2; i++) {
        CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&event[i], cudaEventDisableTiming | cudaEventBlockingSync));
    }


    u32 const search_index_buffer_length = grid_size * block_size * 3;

    u32 const search_index_buffer_size    = search_index_buffer_length * sizeof(u32);
    u32 const has_improvement_buffer_size = grid_size * block_size;
    u32 const payload_buffer_size         = grid_size * block_size * 64;

    u8  *payload_host;
    u32 *search_index_host;
    u8  *has_improvement_host;

    u8  *payload_device;
    u32 *search_index_device;
    u8  *has_improvement_device;

    CHECK_CUDA_ERROR(cudaMallocHost(&payload_host,      2 * payload_buffer_size));
    CHECK_CUDA_ERROR(cudaMallocHost(&search_index_host, 2 * search_index_buffer_size));
    CHECK_CUDA_ERROR(cudaMallocHost(&has_improvement_host,          2 * has_improvement_buffer_size));

    CHECK_CUDA_ERROR(cudaMalloc(&payload_device,      2 * payload_buffer_size));
    CHECK_CUDA_ERROR(cudaMalloc(&search_index_device, 2 * search_index_buffer_size));
    CHECK_CUDA_ERROR(cudaMalloc(&has_improvement_device,          2 * has_improvement_buffer_size));

    u32 buf_idx = 0;

    for (u32 i = 0; i < jobs[0].size(); i++) {
        generator.next(jobs[0][i]);
    }

    search({
        .host = {
            .payload      = payload_host,
            .search_index = search_index_host,
            .has_improvement = has_improvement_host,
        },
        .device = {
            .payload      = payload_device,
            .search_index = search_index_device,
            .has_improvement          = has_improvement_device,
        },

        .grid_size  = grid_size,
        .block_size = block_size,

        .zero_block_count = 0,
        .best_block = 0xFFFFFFFF,

        .stream = stream[0],
        .event  = event[0],
    }, jobs[0], header, header_len);

    u8  best_payload[64];
    u32 best_payload_len = 0;
    u32 best_hash[SHA256_STATE_SIZE] = { 
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 
    };

    while (true) {
        auto start = std::chrono::high_resolution_clock::now();

        u32 next_buf_idx = 1 - buf_idx;

        for (u32 i = 0; i < jobs[next_buf_idx].size(); i++) {
            generator.next(jobs[next_buf_idx][i]);
        }

        u32 zero_block_count = 8;
        for (u32 i = 0; i < 8; i++) {
            if (best_hash[i] != 0) {
                zero_block_count = i;
                break;
            }
        }

        u32 best_block = best_hash[zero_block_count];

        search({
            .host = {
                .payload         = payload_host         + next_buf_idx * payload_buffer_size,
                .search_index    = search_index_host    + next_buf_idx * search_index_buffer_length,
                .has_improvement = has_improvement_host + next_buf_idx * has_improvement_buffer_size,
            },
            .device = {
                .payload         = payload_device         + next_buf_idx * payload_buffer_size,
                .search_index    = search_index_device    + next_buf_idx * search_index_buffer_length,
                .has_improvement = has_improvement_device + next_buf_idx * has_improvement_buffer_size,
            },

            .grid_size  = grid_size,
            .block_size = block_size,

            .zero_block_count = zero_block_count,
            .best_block = best_block,

            .stream = stream[next_buf_idx],
            .event  = event[next_buf_idx],
        }, jobs[next_buf_idx], header, header_len);

        CHECK_CUDA_ERROR(cudaEventSynchronize(event[buf_idx]));

        u8 const *has_improvement = has_improvement_host + buf_idx * has_improvement_buffer_size;

        u32 new_hash[8];
        u8 new_payload[64];
        u32 new_payload_len;
        for (int i = 0; i < block_size * grid_size; i++) {
            if (has_improvement[i]) {
                auto job = jobs[buf_idx][i];
                new_payload_len = header_len + job.length;
                find_best_payload(job, header, header_len, new_payload, new_hash);

                if (!is_better_hash(best_hash, new_hash)) {
                    break;
                }

                memcpy(best_hash,    new_hash,    32);
                memcpy(best_payload, new_payload, 64);
                best_payload_len = new_payload_len;

                printf("NEW BEST! %.*s - %08x %08x %08x %08x %08x %08x %08x %08x\n",
                    best_payload_len,
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
        auto job = jobs[buf_idx].back();
        printf("%3.3f GH/s | Last = search_idx: [%2d, %2d, %2d], len: %2d, product_idx: %" PRIu64 "\n", 
            hashrate,
            job.search_idxs[0],
            job.search_idxs[1],
            job.search_idxs[2],
            job.length,
            job.product_idx
            );

        buf_idx = next_buf_idx;
    }

    return 0;
}

