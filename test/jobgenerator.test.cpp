#include "gtest/gtest.h"
#include "gmock/gmock-matchers.h"
#include "jobgenerator.hpp"

TEST(JobGenerator, GeneratesCorrectJobsOfMaxLengthFour) {
    JobGenerator generator(4);

    JobDescription job;

    EXPECT_TRUE(generator.next(job));

    EXPECT_THAT(job.search_idxs, testing::ElementsAre(2, 1, 0));
    EXPECT_EQ(job.length, 3);
    EXPECT_EQ(job.product_idx, 0);

    for (u32 i = 0; i < 64; i++) {
        EXPECT_TRUE(generator.next(job));

        EXPECT_THAT(job.search_idxs, testing::ElementsAre(2, 1, 0));
        EXPECT_EQ(job.length, 4);
        EXPECT_EQ(job.product_idx, i);
    }

    for (u32 i = 0; i < 64; i++) {
        EXPECT_TRUE(generator.next(job));

        EXPECT_THAT(job.search_idxs, testing::ElementsAre(3, 1, 0));
        EXPECT_EQ(job.length, 4);
        EXPECT_EQ(job.product_idx, i);
    }

    for (u32 i = 0; i < 64; i++) {
        EXPECT_TRUE(generator.next(job));

        EXPECT_THAT(job.search_idxs, testing::ElementsAre(3, 2, 0));
        EXPECT_EQ(job.length, 4);
        EXPECT_EQ(job.product_idx, i);
    }

    for (u32 i = 0; i < 64; i++) {
        EXPECT_TRUE(generator.next(job));

        EXPECT_THAT(job.search_idxs, testing::ElementsAre(3, 2, 1));
        EXPECT_EQ(job.length, 4);
        EXPECT_EQ(job.product_idx, i);
    }

    EXPECT_FALSE(generator.next(job));
}

TEST(LongJobGenerator, GeneratesTheExpectedPayloads) {
    LongJobGenerator generator("toteload/davidbos+dot+me/");

    u8 payload[64];
    u8 match_payload[52];

    memcpy(match_payload, "toteload/davidbos+dot+me////////////////////////////", 52);

    // Convert to big endian
    for (u32 i = 0; i < 52 / 4; i++) {
        std::swap(match_payload[i*4+0], match_payload[i*4+3]);
        std::swap(match_payload[i*4+1], match_payload[i*4+2]);
    }

    // The first 52 bytes of the payload should just be the readable part in big endian.
    EXPECT_TRUE(generator.next(payload));
    EXPECT_EQ(memcmp(payload, match_payload, 52), 0);

    EXPECT_EQ(payload[55], 0);
    EXPECT_EQ(payload[54], 0);
    EXPECT_EQ(payload[53], 0);
    EXPECT_EQ(payload[52], 0x80);

    EXPECT_EQ(payload[59], 0);
    EXPECT_EQ(payload[58], 0);
    EXPECT_EQ(payload[57], 0);
    EXPECT_EQ(payload[56], 0);

    EXPECT_EQ(payload[63], 0);
    EXPECT_EQ(payload[62], 0);
    EXPECT_EQ(payload[61], 440 / 256);
    EXPECT_EQ(payload[60], 440 % 256);

    memcpy(match_payload, "toteload/davidbos+dot+me/+//////////////////////////", 52);

    // Convert to big endian
    for (u32 i = 0; i < 52 / 4; i++) {
        std::swap(match_payload[i*4+0], match_payload[i*4+3]);
        std::swap(match_payload[i*4+1], match_payload[i*4+2]);
    }

    // The first 52 bytes of the payload should just be the readable part in big endian.
    EXPECT_TRUE(generator.next(payload));
    EXPECT_EQ(memcmp(payload, match_payload, 52), 0);

    EXPECT_EQ(payload[55], 0);
    EXPECT_EQ(payload[54], 0);
    EXPECT_EQ(payload[53], 0);
    EXPECT_EQ(payload[52], 0x80);

    EXPECT_EQ(payload[59], 0);
    EXPECT_EQ(payload[58], 0);
    EXPECT_EQ(payload[57], 0);
    EXPECT_EQ(payload[56], 0);

    EXPECT_EQ(payload[63], 0);
    EXPECT_EQ(payload[62], 0);
    EXPECT_EQ(payload[61], 440 / 256);
    EXPECT_EQ(payload[60], 440 % 256);
}
