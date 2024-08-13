#include "gtest/gtest.h"
#include "gmock/gmock-matchers.h"
#include <stdint.h>

extern void sha256_block(const uint8_t payload[64], uint32_t state[8]);

TEST(Sha256, ProducesCorrectHash) {
    uint32_t const seed_state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };

    uint32_t state[8];
    memcpy(state, seed_state, 32);

    uint8_t payload[64] = { 0 };
    memcpy(payload, "jffry/21+GHs/1xRTX4090/CanYouHashFaster/AAAAAClvyNYha4f", 55);
    payload[55] = 0x80;
    payload[62] = 440 / 256;
    payload[63] = 440 % 256;

    sha256_block(payload, state);

    EXPECT_THAT(state, testing::ElementsAre(
        0x00000000, 0x00000004, 0x340f267b, 0xa07b90ae,
        0xd63f69da, 0x590f155c, 0x140e7cd9, 0x786d65de
    ));

    memcpy(state, seed_state, 32);
    sha256_block(payload, state);

    EXPECT_THAT(state, testing::ElementsAre(
        0x00000000, 0x00000004, 0x340f267b, 0xa07b90ae,
        0xd63f69da, 0x590f155c, 0x140e7cd9, 0x786d65de
    ));
}

