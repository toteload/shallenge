#include "gtest/gtest.h"
#include "gmock/gmock-matchers.h"
#include "hash_search.cuh"

TEST(Sha256L55, ProducesCorrectHash) {
    u8 payload[64];
    memset(payload, 0, 64);

    char const *str_payload = "jffry/21+GHs/1xRTX4090/CanYouHashFaster/AAAAAClvyNYh";
    memcpy(payload, str_payload, 52);

    payload[55] = 0x80;
    payload[62] = 440 / 256;
    payload[63] = 440 % 256;

    u32 reg[8] = { 
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };

    // Convert to big endian
    for (u32 i = 0; i < 16; i++) {
        std::swap(payload[i*4+0], payload[i*4+3]);
        std::swap(payload[i*4+1], payload[i*4+2]);
    }

    alignas(4) u8 m[64];
    memcpy(m, payload, 64);

    sha256_l55_prepare((u32*)m, reg);

    u32 state_head[] = { 0xffffffff, 0xffffffff, };

    m[55] = 'a';
    m[54] = '4';
    m[53] = 'f';

    sha256_l55((u32*)m, reg, state_head);

    EXPECT_THAT(state_head, testing::ElementsAre(0x00000000, 0x00000004));
}

