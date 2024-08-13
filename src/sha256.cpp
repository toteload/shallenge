#include <stdint.h>

#define ROTLEFT(a,b)  (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

#define EP0(x)  (ROTRIGHT(x, 2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x)  (ROTRIGHT(x, 6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))

#define SIG0(x) (ROTRIGHT(x, 7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

#define ROUND_0(k, i) \
    t1 = h + EP1(e) + CH(e,f,g) + k + m[i]; \
    t2 = EP0(a) + MAJ(a,b,c); \
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;

#define ROUND_1(k, i) \
    x = SIG1(m[(i+14)%16]) + m[(i+9)%16] + SIG0(m[(i+1)%16]) + m[i]; \
    m[i] = x; \
    t1 = h + EP1(e) + CH(e,f,g) + k + x; \
    t2 = EP0(a) + MAJ(a,b,c); \
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;


void sha256_block(const uint8_t payload[64], uint32_t state[8]) {
	uint32_t a, b, c, d, e, f, g, h, t1, t2, x;
    uint32_t m[16];

    for (uint32_t i = 0, j = 0; i < 16; i++) {
		m[i] = (payload[i*4] << 24) | (payload[i*4+1] << 16) | (payload[i*4+2] << 8) | (payload[i*4+3]);
    }

	a = state[0];
	b = state[1];
	c = state[2];
	d = state[3];
	e = state[4];
	f = state[5];
	g = state[6];
	h = state[7];

    ROUND_0(0x428a2f98, 0);
    ROUND_0(0x71374491, 1);
    ROUND_0(0xb5c0fbcf, 2);
    ROUND_0(0xe9b5dba5, 3);
    ROUND_0(0x3956c25b, 4);
    ROUND_0(0x59f111f1, 5);
    ROUND_0(0x923f82a4, 6);
    ROUND_0(0xab1c5ed5, 7);
    ROUND_0(0xd807aa98, 8);
    ROUND_0(0x12835b01, 9);
    ROUND_0(0x243185be,10);
    ROUND_0(0x550c7dc3,11);
    ROUND_0(0x72be5d74,12);
    ROUND_0(0x80deb1fe,13);
    ROUND_0(0x9bdc06a7,14);
    ROUND_0(0xc19bf174,15);

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

	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
	state[5] += f;
	state[6] += g;
	state[7] += h;
}

