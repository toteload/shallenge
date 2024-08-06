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
