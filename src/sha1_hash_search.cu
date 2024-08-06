#include "sha1_hash_search.cuh"

constexpr u32 SHA1_STATE_SIZE = 5;

u8 alphabet[ALPHABET_SIZE] = { '/', '+', };

__constant__ u8 alphabet_lut[ALPHABET_SIZE];
__constant__ u32 sha1_init_state[SHA1_STATE_SIZE];

void write_payload(JobDescription const &job, char const *header, u32 header_len, u8 *payload) {
    memset(payload, 0, 64);

    memcpy(payload, header, header_len);
    
    u32 total_len = header_len + job.length;
    payload[total_len] = 0x80;

    u32 bit_len = total_len * 8;
    payload[62] = bit_len / 256;
    payload[63] = bit_len % 256;

    u64 acc = job.product_idx;

    for (u32 i = 0; i < job.length; i++) {
        if (job.search_idxs[0] == i ||
            job.search_idxs[1] == i ||
            job.search_idxs[2] == i)
        {
            continue;
        }

        payload[header_len + i] = alphabet[acc % 64];

        acc /= 64;
    }
}

void initialize_cuda_constants() {
    u32 i = 2;
    for (u32 j = 0; j < 10; i++, j++) { alphabet[i] = '0' + j; }
    for (u32 j = 0; j < 26; i++, j++) { alphabet[i] = 'a' + j; }
    for (u32 j = 0; j < 26; i++, j++) { alphabet[i] = 'A' + j; }
    cudaMemcpyToSymbol(alphabet_lut, alphabet, sizeof(alphabet), 0, cudaMemcpyDefault);

    u32 init_state[5] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0, };
    cudaMemcpyToSymbol(sha1_init_state, init_state, sizeof(init_state));
}

// Returns true if the candidate hash was better
__inline_hint__ __device__ __host__
bool is_better_hash(u32 best[SHA1_STATE_SIZE], u32 candidate[SHA1_STATE_SIZE]) {
    for (u32 i = 0; i < SHA1_STATE_SIZE; i++) {
        if (best[i] < candidate[i]) {
            return false;
        }

        if (best[i] > candidate[i]) {
            return true;
        }
    }

    return false;
}

__inline_hint__ __device__ 
void sha1(const u8 block[64], u32 state[SHA1_STATE_SIZE]);

__global__ 
void search_block(const u8 payload[64], const u32 *base_idx, u32 *base_out) {
    u32 offset = threadIdx.x + blockIdx.x * blockDim.x;

    u32 const *idx = base_idx + SEARCH_BLOCK_SIZE * offset;
    u32       *out = base_out + SHA1_STATE_SIZE   * offset;

    // Copy the payload
    u8 block[64];
    for (u32 i = 0; i < 64; i++) {
        block[i] = payload[i];
    }

    u32 candidate[SHA1_STATE_SIZE];

    u32 best[SHA1_STATE_SIZE] = { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, };

    for (u32 i0 = 0; i0 < ALPHABET_SIZE; i0++) { block[idx[0]] = alphabet_lut[i0];
    for (u32 i1 = 0; i1 < ALPHABET_SIZE; i1++) { block[idx[1]] = alphabet_lut[i1];
    for (u32 i2 = 0; i2 < ALPHABET_SIZE; i2++) { block[idx[2]] = alphabet_lut[i2];

        for (u32 i = 0 ; i < SHA1_STATE_SIZE; i++) {
            candidate[i] = sha1_init_state[i];
        }

        sha1(block, candidate);

        if (is_better_hash(best, candidate)) {
            for (u32 i = 0; i < SHA1_STATE_SIZE; i++) {
                best[i] = candidate[i];
            }
        }
    } } }

    for (u32 i = 0 ; i < SHA1_STATE_SIZE; i++) {
        out[i] = best[i];
    }
}

/* 
 * SHA-1 hash in C
 * 
 * Copyright (c) 2023 Project Nayuki. (MIT License)
 * https://www.nayuki.io/page/fast-sha1-hash-implementation-in-x86-assembly
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 * - The above copyright notice and this permission notice shall be included in
 *   all copies or substantial portions of the Software.
 * - The Software is provided "as is", without warranty of any kind, express or
 *   implied, including but not limited to the warranties of merchantability,
 *   fitness for a particular purpose and noninfringement. In no event shall the
 *   authors or copyright holders be liable for any claim, damages or other
 *   liability, whether in an action of contract, tort or otherwise, arising from,
 *   out of or in connection with the Software or the use or other dealings in the
 *   Software.
 */

__device__ void sha1(const uint8_t block[64], uint32_t state[5]) {
    #define ROTL32(x, n)  (((0U + (x)) << (n)) | ((x) >> (32 - (n))))  // Assumes that x is uint32_t and 0 < n < 32
	
	#define LOADSCHEDULE(i)  \
		schedule[i] = (uint32_t)block[i * 4 + 0] << 24  \
		            | (uint32_t)block[i * 4 + 1] << 16  \
		            | (uint32_t)block[i * 4 + 2] <<  8  \
		            | (uint32_t)block[i * 4 + 3] <<  0;
	
	#define SCHEDULE(i)  \
		temp = schedule[(i - 3) & 0xF] ^ schedule[(i - 8) & 0xF] ^ schedule[(i - 14) & 0xF] ^ schedule[(i - 16) & 0xF];  \
		schedule[i & 0xF] = ROTL32(temp, 1);
	
	#define ROUND0a(a, b, c, d, e, i)  LOADSCHEDULE(i)  ROUNDTAIL(a, b, e, ((b & c) | (~b & d))         , i, 0x5A827999)
	#define ROUND0b(a, b, c, d, e, i)  SCHEDULE(i)      ROUNDTAIL(a, b, e, ((b & c) | (~b & d))         , i, 0x5A827999)
	#define ROUND1(a, b, c, d, e, i)   SCHEDULE(i)      ROUNDTAIL(a, b, e, (b ^ c ^ d)                  , i, 0x6ED9EBA1)
	#define ROUND2(a, b, c, d, e, i)   SCHEDULE(i)      ROUNDTAIL(a, b, e, ((b & c) ^ (b & d) ^ (c & d)), i, 0x8F1BBCDC)
	#define ROUND3(a, b, c, d, e, i)   SCHEDULE(i)      ROUNDTAIL(a, b, e, (b ^ c ^ d)                  , i, 0xCA62C1D6)
	
	#define ROUNDTAIL(a, b, e, f, i, k)  \
		e = 0U + e + ROTL32(a, 5) + f + UINT32_C(k) + schedule[i & 0xF];  \
		b = ROTL32(b, 30);
	
	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	
	uint32_t schedule[16];
	uint32_t temp;
	ROUND0a(a, b, c, d, e,  0)
	ROUND0a(e, a, b, c, d,  1)
	ROUND0a(d, e, a, b, c,  2)
	ROUND0a(c, d, e, a, b,  3)
	ROUND0a(b, c, d, e, a,  4)
	ROUND0a(a, b, c, d, e,  5)
	ROUND0a(e, a, b, c, d,  6)
	ROUND0a(d, e, a, b, c,  7)
	ROUND0a(c, d, e, a, b,  8)
	ROUND0a(b, c, d, e, a,  9)
	ROUND0a(a, b, c, d, e, 10)
	ROUND0a(e, a, b, c, d, 11)
	ROUND0a(d, e, a, b, c, 12)
	ROUND0a(c, d, e, a, b, 13)
	ROUND0a(b, c, d, e, a, 14)
	ROUND0a(a, b, c, d, e, 15)
	ROUND0b(e, a, b, c, d, 16)
	ROUND0b(d, e, a, b, c, 17)
	ROUND0b(c, d, e, a, b, 18)
	ROUND0b(b, c, d, e, a, 19)
	ROUND1(a, b, c, d, e, 20)
	ROUND1(e, a, b, c, d, 21)
	ROUND1(d, e, a, b, c, 22)
	ROUND1(c, d, e, a, b, 23)
	ROUND1(b, c, d, e, a, 24)
	ROUND1(a, b, c, d, e, 25)
	ROUND1(e, a, b, c, d, 26)
	ROUND1(d, e, a, b, c, 27)
	ROUND1(c, d, e, a, b, 28)
	ROUND1(b, c, d, e, a, 29)
	ROUND1(a, b, c, d, e, 30)
	ROUND1(e, a, b, c, d, 31)
	ROUND1(d, e, a, b, c, 32)
	ROUND1(c, d, e, a, b, 33)
	ROUND1(b, c, d, e, a, 34)
	ROUND1(a, b, c, d, e, 35)
	ROUND1(e, a, b, c, d, 36)
	ROUND1(d, e, a, b, c, 37)
	ROUND1(c, d, e, a, b, 38)
	ROUND1(b, c, d, e, a, 39)
	ROUND2(a, b, c, d, e, 40)
	ROUND2(e, a, b, c, d, 41)
	ROUND2(d, e, a, b, c, 42)
	ROUND2(c, d, e, a, b, 43)
	ROUND2(b, c, d, e, a, 44)
	ROUND2(a, b, c, d, e, 45)
	ROUND2(e, a, b, c, d, 46)
	ROUND2(d, e, a, b, c, 47)
	ROUND2(c, d, e, a, b, 48)
	ROUND2(b, c, d, e, a, 49)
	ROUND2(a, b, c, d, e, 50)
	ROUND2(e, a, b, c, d, 51)
	ROUND2(d, e, a, b, c, 52)
	ROUND2(c, d, e, a, b, 53)
	ROUND2(b, c, d, e, a, 54)
	ROUND2(a, b, c, d, e, 55)
	ROUND2(e, a, b, c, d, 56)
	ROUND2(d, e, a, b, c, 57)
	ROUND2(c, d, e, a, b, 58)
	ROUND2(b, c, d, e, a, 59)
	ROUND3(a, b, c, d, e, 60)
	ROUND3(e, a, b, c, d, 61)
	ROUND3(d, e, a, b, c, 62)
	ROUND3(c, d, e, a, b, 63)
	ROUND3(b, c, d, e, a, 64)
	ROUND3(a, b, c, d, e, 65)
	ROUND3(e, a, b, c, d, 66)
	ROUND3(d, e, a, b, c, 67)
	ROUND3(c, d, e, a, b, 68)
	ROUND3(b, c, d, e, a, 69)
	ROUND3(a, b, c, d, e, 70)
	ROUND3(e, a, b, c, d, 71)
	ROUND3(d, e, a, b, c, 72)
	ROUND3(c, d, e, a, b, 73)
	ROUND3(b, c, d, e, a, 74)
	ROUND3(a, b, c, d, e, 75)
	ROUND3(e, a, b, c, d, 76)
	ROUND3(d, e, a, b, c, 77)
	ROUND3(c, d, e, a, b, 78)
	ROUND3(b, c, d, e, a, 79)
	
	state[0] = 0U + state[0] + a;
	state[1] = 0U + state[1] + b;
	state[2] = 0U + state[2] + c;
	state[3] = 0U + state[3] + d;
	state[4] = 0U + state[4] + e;
	
	#undef ROTL32
	#undef LOADSCHEDULE
	#undef SCHEDULE
	#undef ROUND0a
	#undef ROUND0b
	#undef ROUND1
	#undef ROUND2
	#undef ROUND3
	#undef ROUNDTAIL
}
