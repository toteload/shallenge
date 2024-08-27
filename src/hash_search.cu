#include "hash_search.cuh"
#include "jobgenerator.hpp"

__constant__ u8 alphabet_lut[ALPHABET_SIZE];

void initialize_cuda_constants() {
    cudaMemcpyToSymbol(alphabet_lut, alphabet, sizeof(alphabet), 0, cudaMemcpyDefault);
}

// Returns true if the candidate hash was better
__device__ __host__
bool is_better_hash(u32 const best[SHA256_STATE_SIZE], u32 const candidate[SHA256_STATE_SIZE]) {
    for (u32 i = 0; i < SHA256_STATE_SIZE; i++) {
        if (best[i] < candidate[i]) {
            return false;
        }

        if (best[i] > candidate[i]) {
            return true;
        }
    }

    return false;
}

__device__ void sha256(const u8 payload[64], u32 state[8]);

__global__ 
void search_block(const u8 *base_payload, const u32 *base_idx, u32 *base_out) {
    u32 offset = threadIdx.x + blockIdx.x * blockDim.x;

    u8  const *payload = base_payload + 64 * offset;
    u32 const *idx     = base_idx     + SEARCH_BLOCK_SIZE * offset;
    u32       *out     = base_out     + 2 * offset;

    u8 block[64];
    memcpy(block, payload, 64);

    u32 best_hash_head[2] = { 0xFFFFFFFF, 0xFFFFFFFF, };

    for (u32 i0 = 0; i0 < ALPHABET_SIZE; i0++) { block[idx[0]] = alphabet_lut[i0];
    for (u32 i1 = 0; i1 < ALPHABET_SIZE; i1++) { block[idx[1]] = alphabet_lut[i1];
    for (u32 i2 = 0; i2 < ALPHABET_SIZE; i2++) { block[idx[2]] = alphabet_lut[i2];

        u32 m[16], candidate[2];
        memcpy(m, block, 64);
        sha256(m, candidate);

        if (is_better_hash(best, candidate)) {
            memcpy(best_hash_head, candidate, 2 * sizeof(u32));
        }
    } } }

    memcpy(out, best_hash_head, 2 * sizeof(u32));
}

#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x)     (ROTRIGHT(x, 2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x)     (ROTRIGHT(x, 6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x)    (ROTRIGHT(x, 7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x)    (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

#define ROUND_0(k, x) \
    t1 = h + EP1(e) + CH(e,f,g) + k + x; \
    t2 = EP0(a) + MAJ(a,b,c); \
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;

#define ROUND_1(k, i) \
    x = SIG1(m[(i+14)%16]) + m[(i+9)%16] + SIG0(m[(i+1)%16]) + m[i]; \
    m[i] = x; \
    ROUND_0(k, x)

__device__ void sha256(u32 m[16], u32 state_head_out[2]) {
	u32 a = 0x6a09e667;
	u32 b = 0xbb67ae85;
	u32 c = 0x3c6ef372;
	u32 d = 0xa54ff53a;

	u32 e = 0x510e527f;
	u32 f = 0x9b05688c;
	u32 g = 0x1f83d9ab;
	u32 h = 0x5be0cd19;

	u32 t1, t2, x; 

    ROUND_0(0x428a2f98,m[ 0]);
    ROUND_0(0x71374491,m[ 1]);
    ROUND_0(0xb5c0fbcf,m[ 2]);
    ROUND_0(0xe9b5dba5,m[ 3]);
    ROUND_0(0x3956c25b,m[ 4]);
    ROUND_0(0x59f111f1,m[ 5]);
    ROUND_0(0x923f82a4,m[ 6]);
    ROUND_0(0xab1c5ed5,m[ 7]);
    ROUND_0(0xd807aa98,m[ 8]);
    ROUND_0(0x12835b01,m[ 9]);
    ROUND_0(0x243185be,m[10]);
    ROUND_0(0x550c7dc3,m[11]);
    ROUND_0(0x72be5d74,m[12]);
    ROUND_0(0x80deb1fe,m[13]);
    ROUND_0(0x9bdc06a7,m[14]);
    ROUND_0(0xc19bf174,m[15]);

    ROUND_1(0xe49b69c1, 0);
    ROUND_1(0xefbe4786, 1);
    ROUND_1(0x0fc19dc6, 2);
    ROUND_1(0x240ca1cc, 3);
    ROUND_1(0x2de92c6f, 4);
    ROUND_1(0x4a7484aa, 5);
    ROUND_1(0x5cb0a9dc, 6);
    ROUND_1(0x76f988da, 7);
    ROUND_1(0x983e5152, 8);
    ROUND_1(0xa831c66d, 9);
    ROUND_1(0xb00327c8,10);
    ROUND_1(0xbf597fc7,11);
    ROUND_1(0xc6e00bf3,12);
    ROUND_1(0xd5a79147,13);
    ROUND_1(0x06ca6351,14);
    ROUND_1(0x14292967,15);

    ROUND_1(0x27b70a85, 0);
    ROUND_1(0x2e1b2138, 1);
    ROUND_1(0x4d2c6dfc, 2);
    ROUND_1(0x53380d13, 3);
    ROUND_1(0x650a7354, 4);
    ROUND_1(0x766a0abb, 5);
    ROUND_1(0x81c2c92e, 6);
    ROUND_1(0x92722c85, 7);
    ROUND_1(0xa2bfe8a1, 8);
    ROUND_1(0xa81a664b, 9);
    ROUND_1(0xc24b8b70,10);
    ROUND_1(0xc76c51a3,11);
    ROUND_1(0xd192e819,12);
    ROUND_1(0xd6990624,13);
    ROUND_1(0xf40e3585,14);
    ROUND_1(0x106aa070,15);

    ROUND_1(0x19a4c116, 0);
    ROUND_1(0x1e376c08, 1);
    ROUND_1(0x2748774c, 2);
    ROUND_1(0x34b0bcb5, 3);
    ROUND_1(0x391c0cb3, 4);
    ROUND_1(0x4ed8aa4a, 5);
    ROUND_1(0x5b9cca4f, 6);
    ROUND_1(0x682e6ff3, 7);
    ROUND_1(0x748f82ee, 8);
    ROUND_1(0x78a5636f, 9);
    ROUND_1(0x84c87814,10);
    ROUND_1(0x8cc70208,11);
    ROUND_1(0x90befffa,12);
    ROUND_1(0xa4506ceb,13);
    ROUND_1(0xbef9a3f7,14);
    ROUND_1(0xc67178f2,15);

	state_head[0] = 0x6a09e667a + a;
	state_head[1] = 0xbb67ae85b + b;
}

__host__
void sha256_host(const u8 payload[64], u32 state[8])
{
    static const u32 ROUND_CONSTANT[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };

	u32 a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

	for (i = 0, j = 0; i < 16; ++i, j += 4)
		m[i] = (payload[j] << 24) | (payload[j + 1] << 16) | (payload[j + 2] << 8) | (payload[j + 3]);
	for ( ; i < 64; ++i)
		m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

	a = state[0];
	b = state[1];
	c = state[2];
	d = state[3];
	e = state[4];
	f = state[5];
	g = state[6];
	h = state[7];

	for (i = 0; i < 64; ++i) {
		t1 = h + EP1(e) + CH(e,f,g) + ROUND_CONSTANT[i] + m[i];
		t2 = EP0(a) + MAJ(a,b,c);
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
	state[5] += f;
	state[6] += g;
	state[7] += h;
}

