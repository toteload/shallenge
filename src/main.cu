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
        u32 *out;
    } host;

    struct {
        u8  *payload;
        u32 *search_index;
        u32 *out;
    } device;

    int grid_size;
    int block_size;

    cudaStream_t stream;
    cudaEvent_t  event;
};

void search(Context const &ctx, std::vector<JobDescription> &jobs, char const *header, u32 header_len) {
    u32 const payload_buffer_size      = ctx.grid_size * 64;
    u32 const search_index_buffer_size = ctx.grid_size * 12;
    u32 const out_buffer_size          = ctx.grid_size * 32;

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

    search_block2<<<ctx.grid_size, ctx.block_size, 0, ctx.stream>>>(
        ctx.device.payload,
        ctx.device.search_index,
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

void find_payload(JobDescription const &job, char const *header, u32 header_len, u32 const needle[8], u8 payload[64]) {
    write_payload(job, header, header_len, payload);

    for (u32 i0 = 0; i0 < ALPHABET_SIZE; i0++) { payload[header_len + job.search_idxs[0]] = alphabet[i0];
    for (u32 i1 = 0; i1 < ALPHABET_SIZE; i1++) { payload[header_len + job.search_idxs[1]] = alphabet[i1];
    for (u32 i2 = 0; i2 < ALPHABET_SIZE; i2++) { payload[header_len + job.search_idxs[2]] = alphabet[i2];
        u32 candidate[SHA256_STATE_SIZE] = { 
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
        };

        sha256_host(payload, candidate);

        if (memcmp(needle, candidate, 32) == 0) {
            return;
        }
    } } }
}

int main() {
    initialize_cuda_constants();

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

    int block_size = 64;
    int grid_size = 65535;

    std::vector<JobDescription> jobs[2];
    jobs[0].resize(grid_size);
    jobs[1].resize(grid_size);

    cudaStream_t stream[2];
    for (u32 i = 0; i < 2; i++) {
        CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));
    }

    cudaEvent_t event[2];
    for (u32 i = 0; i < 2; i++) {
        CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&event[i], cudaEventDisableTiming | cudaEventBlockingSync));
    }

    u32 const payload_buffer_size = grid_size * 64;

    u32 const search_index_buffer_length = grid_size * 3;
    u32 const out_buffer_length          = grid_size * SHA256_STATE_SIZE;

    u32 const search_index_buffer_size = search_index_buffer_length * sizeof(u32);
    u32 const out_buffer_size          = out_buffer_length          * sizeof(u32);

    u8  *payload_host;
    u32 *search_index_host;
    u32 *out_host;

    u8  *payload_device;
    u32 *search_index_device;
    u32 *out_device;

    CHECK_CUDA_ERROR(cudaMallocHost(&payload_host,      2 * payload_buffer_size));
    CHECK_CUDA_ERROR(cudaMallocHost(&search_index_host, 2 * search_index_buffer_size));
    CHECK_CUDA_ERROR(cudaMallocHost(&out_host,          2 * out_buffer_size));

    CHECK_CUDA_ERROR(cudaMalloc(&payload_device,      2 * payload_buffer_size));
    CHECK_CUDA_ERROR(cudaMalloc(&search_index_device, 2 * search_index_buffer_size));
    CHECK_CUDA_ERROR(cudaMalloc(&out_device,          2 * out_buffer_size));

    u32 buf_idx = 0;

    for (u32 i = 0; i < jobs[0].size(); i++) {
        generator.next(jobs[0][i]);
    }

    search({
        .host = {
            .payload      = payload_host,
            .search_index = search_index_host,
            .out          = out_host,
        },
        .device = {
            .payload      = payload_device,
            .search_index = search_index_device,
            .out          = out_device,
        },

        .grid_size  = grid_size,
        .block_size = block_size,

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

        search({
            .host = {
                .payload      = payload_host      + next_buf_idx * payload_buffer_size,
                .search_index = search_index_host + next_buf_idx * search_index_buffer_length,
                .out          = out_host          + next_buf_idx * out_buffer_length,
            },
            .device = {
                .payload      = payload_device      + next_buf_idx * payload_buffer_size,
                .search_index = search_index_device + next_buf_idx * search_index_buffer_length,
                .out          = out_device          + next_buf_idx * out_buffer_length,
            },

            .grid_size  = grid_size,
            .block_size = block_size,

            .stream = stream[next_buf_idx],
            .event  = event[next_buf_idx],
        }, jobs[next_buf_idx], header, header_len);

        CHECK_CUDA_ERROR(cudaEventSynchronize(event[buf_idx]));

        u32 const *out = out_host + buf_idx * out_buffer_length;

        for (int i = 0; i < jobs[0].size(); i++) {
            u32 const *candidate = out + i * SHA256_STATE_SIZE;
            if (is_better_hash(best_hash, candidate)) {
                auto job = jobs[buf_idx][i];
                memcpy(best_hash, candidate, 32);
                best_payload_len = header_len + job.length;
                find_payload(job, header, header_len, candidate, best_payload);

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

        double hashrate = (double)(64*64*64) * grid_size / 1'000'000'000.0 / d.count();
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

