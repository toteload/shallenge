#include "gtest/gtest.h"
#include "gmock/gmock-matchers.h"
#include <stdint.h>

extern void sha256_l55_prepare(uint32_t m[16], uint32_t reg[8]);
extern void sha256_l55(uint32_t m[16], uint32_t reg[8], uint32_t state_head[2]);

TEST(Sha256, ProducesCorrectHash) {
    uint32_t const seed_state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };

    char const *str_payload = "jffry/21+GHs/1xRTX4090/CanYouHashFaster/AAAAAClvyNYha4f";
    uint8_t payload[64];
    memset(payload, 0, 64);
    memcpy(payload, str_payload, 55);
    payload[55] = 0x80;
    payload[62] = 440 / 256;
    payload[63] = 440 % 256;

    uint32_t m[16], reg[8];

    for (uint32_t i = 0; i < 16; i++) {
        m[i] = (payload[i*4] << 24) | (payload[i*4 + 1] << 16) | (payload[i*4 + 2] << 8) | (payload[i*4 + 3]);
    }

    memcpy(reg, seed_state, 32);

    sha256_l55_prepare(m, reg);

    uint32_t state_head = { 0xffffffff, 0xffffffff, };

    sha256_l55(m, reg, state_head);

    EXPECT_THAT(state_head, testing::ElementsAre(0x00000000, 0x00000004));
}

