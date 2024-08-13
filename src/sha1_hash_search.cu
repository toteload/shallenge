#include "sha1_hash_search.cuh"

u8 alphabet[ALPHABET_SIZE] = { '/', '+', };

__constant__ u8 alphabet_lut[ALPHABET_SIZE];

void write_payload(JobDescription const &job, char const *header, u32 header_len, u8 payload[64]) {
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
    u32       *out     = base_out     + SHA256_STATE_SIZE * offset;

    u8 block[64];
    memcpy(block, payload, 64);

    u32 best[SHA256_STATE_SIZE] = { 
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 
    };

    for (u32 i0 = 0; i0 < ALPHABET_SIZE; i0++) { block[idx[0]] = alphabet_lut[i0];
    for (u32 i1 = 0; i1 < ALPHABET_SIZE; i1++) { block[idx[1]] = alphabet_lut[i1];
    for (u32 i2 = 0; i2 < ALPHABET_SIZE; i2++) { block[idx[2]] = alphabet_lut[i2];
        u32 candidate[SHA256_STATE_SIZE] = { 
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
        };

        sha256(block, candidate);

        if (is_better_hash(best, candidate)) {
            memcpy(best, candidate, 32);
        }
    } } }

    memcpy(out, best, 32);
}

#define ROTLEFT(a,b)  (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x)     (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x)     (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x)    (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x)    (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

#define ROUND(k, i) \
    t1 = h + EP1(e) + CH(e,f,g) + k + m[i]; t2 = EP0(a) + MAJ(a,b,c); \
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;

__device__ void sha256(const u8 payload[64], u32 state[8]) {
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

    ROUND(0x428a2f98, 0);
    ROUND(0x71374491, 1);
    ROUND(0xb5c0fbcf, 2);
    ROUND(0xe9b5dba5, 3);
    ROUND(0x3956c25b, 4);
    ROUND(0x59f111f1, 5);
    ROUND(0x923f82a4, 6);
    ROUND(0xab1c5ed5, 7);
    ROUND(0xd807aa98, 8);
    ROUND(0x12835b01, 9);
    ROUND(0x243185be,10);
    ROUND(0x550c7dc3,11);
    ROUND(0x72be5d74,12);
    ROUND(0x80deb1fe,13);
    ROUND(0x9bdc06a7,14);
    ROUND(0xc19bf174,15);
    ROUND(0xe49b69c1,16);
    ROUND(0xefbe4786,17);
    ROUND(0x0fc19dc6,18);
    ROUND(0x240ca1cc,19);
    ROUND(0x2de92c6f,20);
    ROUND(0x4a7484aa,21);
    ROUND(0x5cb0a9dc,22);
    ROUND(0x76f988da,23);
    ROUND(0x983e5152,24);
    ROUND(0xa831c66d,25);
    ROUND(0xb00327c8,26);
    ROUND(0xbf597fc7,27);
    ROUND(0xc6e00bf3,28);
    ROUND(0xd5a79147,29);
    ROUND(0x06ca6351,30);
    ROUND(0x14292967,31);
    ROUND(0x27b70a85,32);
    ROUND(0x2e1b2138,33);
    ROUND(0x4d2c6dfc,34);
    ROUND(0x53380d13,35);
    ROUND(0x650a7354,36);
    ROUND(0x766a0abb,37);
    ROUND(0x81c2c92e,38);
    ROUND(0x92722c85,39);
    ROUND(0xa2bfe8a1,40);
    ROUND(0xa81a664b,41);
    ROUND(0xc24b8b70,42);
    ROUND(0xc76c51a3,43);
    ROUND(0xd192e819,44);
    ROUND(0xd6990624,45);
    ROUND(0xf40e3585,46);
    ROUND(0x106aa070,47);
    ROUND(0x19a4c116,48);
    ROUND(0x1e376c08,49);
    ROUND(0x2748774c,50);
    ROUND(0x34b0bcb5,51);
    ROUND(0x391c0cb3,52);
    ROUND(0x4ed8aa4a,53);
    ROUND(0x5b9cca4f,54);
    ROUND(0x682e6ff3,55);
    ROUND(0x748f82ee,56);
    ROUND(0x78a5636f,57);
    ROUND(0x84c87814,58);
    ROUND(0x8cc70208,59);
    ROUND(0x90befffa,60);
    ROUND(0xa4506ceb,61);
    ROUND(0xbef9a3f7,62);
    ROUND(0xc67178f2,63);

	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
	state[5] += f;
	state[6] += g;
	state[7] += h;
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

